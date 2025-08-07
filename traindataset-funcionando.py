import os
import json
import gc
import psutil
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback
)

# Usa apenas CPU
device = torch.device("cpu")

# Carrega modelo e processador
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Gradient checkpointing para economizar RAM
model.gradient_checkpointing_enable()

# Garante que o modelo esteja na CPU
model.to(device)

# Configuração de tokens especiais
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = "<pad>"
model.config.pad_token_id = processor.tokenizer.pad_token_id
if model.config.decoder_start_token_id is None:
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")

# Função que transforma JSON em string
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

# Dataset personalizado
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
            print(f"Erro ao processar imagem: {img_path} | {e}")
            pixel_values = torch.zeros((3, 480, 640))

        try:
            with open(json_path, "r", encoding="latin1") as f:
                json_data = json.load(f)
            target_text = json_para_string(json_data)
        except Exception as e:
            print(f"Erro ao ler {json_path}: {e}")
            target_text = ""

        input_ids = self.processor.tokenizer(
            target_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "labels": input_ids
        }

# Monitor de memória via callback
class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, every_n_steps=10):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / (1024 ** 2)  # Em MB
            print(f"🧠 [Passo {state.global_step}] Memória RAM usada: {mem_mb:.2f} MB")

# Caminhos
image_dir = "dataset_receitas/images"
json_dir = "dataset_receitas/annotations"

# Parâmetros
batch_size = 1
num_epochs = 10

# Dataset
dataset = ReceitaDataset(image_dir, json_dir, processor)
num_examples = len(dataset)
max_steps = (num_examples * num_epochs) // batch_size
print(f"📦 Dataset carregado com {num_examples} exemplos. Max steps: {max_steps}")

# Divisão treino/validação
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Argumentos do treinamento
args = Seq2SeqTrainingArguments(
    output_dir="./donut-receitas",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    dataloader_num_workers=0,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=False,  # CPU não suporta float16
    eval_strategy="no",
    max_steps=max_steps
)

# Cria trainer com o monitor de RAM
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[MemoryMonitorCallback(every_n_steps=5)]
)

# Inicia o treinamento
trainer.train()

# Libera memória depois
gc.collect()

# Salva modelo e tokenizer
model.save_pretrained("./donut-receitas")
processor.tokenizer.save_pretrained("./donut-receitas")

print("✅ Treinamento concluído e modelo salvo com sucesso!")
