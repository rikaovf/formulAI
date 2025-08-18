# 🔧 Instalação de dependências (apenas no Colab ou ambientes limpos)
try:
    import transformers
except ImportError:
    import os
    os.system("pip install -U transformers accelerate psutil pillow")

import os
import json
import gc
import psutil
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback
)
from difflib import SequenceMatcher

# 🚀 Usa GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Dispositivo em uso: {device}")

# 🔽 Carrega modelo e processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# ⚙️ Configurações
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = "<pad>"

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")
model.config.eos_token_id = processor.tokenizer.eos_token_id

model.to(device)
model.gradient_checkpointing_enable()

# 🔤 Conversão JSON → string
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

# 📦 Dataset personalizado
class ReceitaDataset(Dataset):
    def __init__(self, img_dir, json_dir, processor, max_length=512):
        self.img_dir = img_dir
        self.json_dir = json_dir
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

        input_ids = self.processor.tokenizer(
            target_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).input_ids[0]

        return {"pixel_values": pixel_values, "labels": input_ids}

# 🧩 Data collator
def data_collator(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
    return {"pixel_values": pixel_values, "labels": labels}

# 🧠 Callback de memória
class MemoriaCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 5 == 0:
            processo = psutil.Process(os.getpid())
            mem = processo.memory_info().rss / (1024 ** 2)
            print(f"🧠 Passo {state.global_step}: RAM usada {mem:.2f} MB")

    def on_epoch_end(self, args, state, control, **kwargs):
        gc.collect()
        print("🧹 Memória liberada no fim da época")

# 📏 Similaridade (avaliação)
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
            saida_ids = model.generate(pixel_values, max_new_tokens=512)
            predicao = processor.batch_decode(saida_ids, skip_special_tokens=True)[0]
            sim = calcular_similaridade(predicao, rotulo)
            print(f"\n🔍 Amostra {i+1}:")
            print(f"✔️ Esperado: {rotulo}")
            print(f"🤖 Predito : {predicao}")
            print(f"📏 Similaridade: {sim:.2f}")
    model.train()

# 📁 Diretórios do dataset (ajuste se necessário)
image_dir = "dataset_receitas/images"
json_dir = "dataset_receitas/annotations"

# ⚙️ Hiperparâmetros
batch_size = 1
num_epochs = 5
max_length = 512

# 📊 Dataset
dataset = ReceitaDataset(image_dir, json_dir, processor, max_length=max_length)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, val_size])

max_steps = (len(train_dataset) * num_epochs) // batch_size
print(f"📦 Dataset com {len(dataset)} exemplos | Max steps: {max_steps}")

# 🔧 Argumentos do treino
args = Seq2SeqTrainingArguments(
    output_dir="./donut-receitas",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    dataloader_num_workers=2,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    report_to="none"
)

# 🧠 Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
    callbacks=[MemoriaCallback()]
)

# 🚀 Início do treinamento
trainer.train()

# 🔍 Pós-treinamento
avaliar_amostras(model, eval_dataset, processor, n=3)

# 💾 Salvamento
model.save_pretrained("./donut-receitas")
processor.save_pretrained("./donut-receitas")

print("✅ Treinamento concluído e modelo salvo com sucesso!")
