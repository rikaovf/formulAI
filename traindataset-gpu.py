# ====================================================
# 🚀 Treinamento Donut com validações (v2 — compat HF)
# ====================================================
# Principais mudanças:
# - Compatibilidade com versões antigas do transformers: usa evaluation_strategy
#   e faz fallback para eval_strategy quando necessário (erro que você reportou).
# - Pad token adicionado ANTES, e resize dos embeddings feito após QUALQUER
#   alteração de vocabulário (evita desalinhamento de IDs).
# - Data collator mascara PAD com -100 (ignora no cálculo de loss),
#   reduzindo repetição e colapso na geração.
# - remove_unused_columns=False (evita o Trainer descartar 'pixel_values').
# - Geração com no_repeat_ngram_size e repetition_penalty.

import os
import json
import gc
import psutil
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback
)
from difflib import SequenceMatcher
from packaging import version
import transformers as hf

# ====================================================
# ⚙️ Dispositivo
# ====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Dispositivo em uso: {device}")
print(f"🔧 transformers versão: {hf.__version__}")

# ====================================================
# 🔽 Modelo e Processor
# ====================================================
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# ====================================================
# 📌 Tokens Especiais (adicione PAD antes de tudo)
# ====================================================
if processor.tokenizer.pad_token is None:
    processor.tokenizer.add_special_tokens({'pad_token': "<pad>"})

special_tokens = [
    "<s_receita>", "</s_receita>",
    "<s_medico>", "</s_medico>",
    "<s_crm>", "</s_crm>",
    "<s_paciente>", "</s_paciente>",
    "<s_tipo>", "</s_tipo>",
    "<s_posologia>", "</s_posologia>",
    "<s_componentes>", "</s_componentes>",
    "<s_componente>", "</s_componente>",
    "<s_dosagem>", "</s_dosagem>",
    "<s_unidade>", "</s_unidade>"
]

num_added_tokens = processor.tokenizer.add_tokens(special_tokens)
print(f"✅ {num_added_tokens} tokens especiais adicionados.")

# Redimensiona embeddings no decoder DEPOIS de qualquer alteração no vocab
model.decoder.resize_token_embeddings(len(processor.tokenizer))

# ⚙️ IDs de tokens especiais
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_receita>")
model.config.eos_token_id = processor.tokenizer.convert_tokens_to_ids("</s_receita>")

model.to(device)
model.gradient_checkpointing_enable()

# ====================================================
# 🔍 Validação de Tokens e Embeddings
# ====================================================
print("\n🔎 Validação dos tokens especiais...")
for t in special_tokens + [processor.tokenizer.pad_token]:
    tok_id = processor.tokenizer.convert_tokens_to_ids(t)
    if tok_id == processor.tokenizer.unk_token_id:
        raise ValueError(f"❌ Token {t} não foi adicionado corretamente!")
    else:
        print(f"✅ Token {t} → id {tok_id}")

print("\n🔎 Verificando alinhamento de config...")
assert model.config.pad_token_id == processor.tokenizer.pad_token_id, "❌ Pad token desalinhado!"
print("✅ pad_token_id correto.")
print("Decoder start:", processor.tokenizer.decode([model.config.decoder_start_token_id]))
print("EOS token:", processor.tokenizer.decode([model.config.eos_token_id]))

# ====================================================
# 🔤 Conversão JSON → String
# ====================================================

def json_para_string(dados):
    out = "<s_receita>"
    out += f"<s_medico>{dados['medico']}</s_medico>"
    out += f"<s_crm>{dados['crm']}</s_crm>"
    out += f"<s_paciente>{dados['paciente']}</s_paciente>"
    out += f"<s_tipo>{dados['tipo']}</s_tipo>"
    out += f"<s_posologia>{dados['posologia']}</s_posologia>"
    for p in dados['componentes']:
        out += "<s_componentes>"
        out += f"<s_componente>{p['componente']}</s_componente>"
        out += f"<s_dosagem>{p['dosagem']}</s_dosagem>"
        out += f"<s_unidade>{p['unidade']}</s_unidade>"
        out += "</s_componentes>"
    out += "</s_receita>"
    return out

# ====================================================
# 📦 Dataset
# ====================================================

class ReceitaDataset(Dataset):
    def __init__(self, img_dir, json_dir, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
        self.examples = [
            (os.path.join(img_dir, f), os.path.join(json_dir, f.replace(".jpg", ".json")))
            for f in os.listdir(img_dir)
            if f.endswith(".jpg") and os.path.exists(os.path.join(json_dir, f.replace(".jpg", ".json")))
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        img_path, json_path = self.examples[idx]

        try:
            image = Image.open(img_path).convert("RGB").resize((640, 480))
            pixel_values = self.processor(image, return_tensors="pt").pixel_values[0]
        except Exception as e:
            print(f"Erro na imagem: {img_path} | {e}")
            pixel_values = torch.zeros((3, 480, 640))

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            target_text = json_para_string(json_data)
        except Exception as e:
            print(f"Erro no JSON: {json_path} | {e}")
            target_text = ""

        input_ids = processor.tokenizer(
            target_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).input_ids[0]

        return {"pixel_values": pixel_values, "labels": input_ids}

# ====================================================
# 🧩 Data collator (mascara PAD como -100)
# ====================================================

def data_collator(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
    pad_id = processor.tokenizer.pad_token_id
    labels[labels == pad_id] = -100
    return {"pixel_values": pixel_values, "labels": labels}

# ====================================================
# 🧠 Callback de memória
# ====================================================

class MemoriaCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            processo = psutil.Process(os.getpid())
            mem = processo.memory_info().rss / (1024 ** 2)
            print(f"🧠 Passo {state.global_step}: RAM usada {mem:.2f} MB")

    def on_epoch_end(self, args, state, control, **kwargs):
        gc.collect()
        print("🧹 Memória liberada no fim da época")

# ====================================================
# 📏 Similaridade (avaliação)
# ====================================================

def calcular_similaridade(pred, label):
    return SequenceMatcher(None, pred, label).ratio()


def avaliar_amostras(model, dataset, processor, n=3):
    model.eval()
    for i in range(min(n, len(dataset))):
        entrada = dataset[i]
        with torch.no_grad():
            pixel_values = entrada["pixel_values"].unsqueeze(0).to(device)
            rotulo = processor.tokenizer.decode(
                entrada["labels"], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            saida_ids = model.generate(
                pixel_values,
                max_new_tokens=512,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                early_stopping=True
            )
            predicao = processor.batch_decode(saida_ids, skip_special_tokens=True)[0]
            sim = calcular_similaridade(predicao, rotulo)
            print(f"\n🔍 Amostra {i+1}:")
            print(f"✔️ Esperado: {rotulo}")
            print(f"🤖 Predito : {predicao}")
            print(f"📏 Similaridade: {sim:.2f}")
    model.train()

# ====================================================
# 📁 Dataset
# ====================================================

image_dir = "dataset_receitas/images"
json_dir = "dataset_receitas/annotations"

batch_size = 1
num_epochs = 5
max_length = 512

processor.feature_extractor.size = {"height": 480, "width": 640}
processor.feature_extractor.do_resize = True

dataset = ReceitaDataset(image_dir, json_dir, processor, max_length=max_length)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, val_size])

print(f"📦 Dataset com {len(dataset)} exemplos")

# 🔍 Validação dataset antes do treino
loader = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)
lbatch = next(iter(loader))
print("\n🔎 Validação dataset:")
print("Pixel values:", lbatch["pixel_values"].shape)
print("Labels:", lbatch["labels"].shape)
print("Exemplo decodificado:", processor.tokenizer.decode(
    torch.where(lbatch["labels"][0] == -100, torch.tensor(processor.tokenizer.pad_token_id), lbatch["labels"][0]),
    skip_special_tokens=False
))

# ====================================================
# 🔧 Treinamento (com fallback de argumentos)
# ====================================================

common_args = dict(
    output_dir="./donut-receitas",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=4,  # simula batch_size=4
    dataloader_num_workers=2,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    report_to="none",
    remove_unused_columns=False,
)

# Tenta usar evaluation_strategy (versões novas). Se falhar, usa eval_strategy (antigas)
try:
    args = Seq2SeqTrainingArguments(
        evaluation_strategy="epoch",
        **common_args
    )
except TypeError:
    print("⚠️ Seq2SeqTrainingArguments não aceita 'evaluation_strategy'. Usando 'eval_strategy'.")
    args = Seq2SeqTrainingArguments(
        eval_strategy="epoch",
        **common_args
    )

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
    callbacks=[MemoriaCallback()]
)

# 🚀 Treinamento
trainer.train()

# ====================================================
# 🔍 Pós-treinamento
# ====================================================
avaliar_amostras(model, eval_dataset, processor, n=3)

# 💾 Salvamento
# salvar direto na pasta do Drive
#Configurado para salvar no drive.
model.save_pretrained("/content/drive/MyDrive/donut-receitas")
processor.save_pretrained("/content/drive/MyDrive/donut-receitas")

print("✅ Modelo e processor salvos no Google Drive com sucesso!")

#model.save_pretrained("./donut-receitas")
#processor.save_pretrained("./donut-receitas")


#print("✅ Treinamento concluído e modelo salvo com sucesso!")
