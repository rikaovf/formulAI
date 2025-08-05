from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from PIL import Image
import torch
import json
import os
import gc
from tqdm import tqdm

# 1. Carrega o modelo e processador
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Se o tokenizer não tiver pad_token, defina-o manualmente
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = "<pad>"

# Agora, defina o pad_token_id no modelo
model.config.pad_token_id = processor.tokenizer.pad_token_id

# Definir o decoder_start_token_id
if model.config.decoder_start_token_id is None:
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")

# 2. Conversão do JSON para string formatada
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

# 3. Carrega dados do diretório
def carregar_dados(img_dir, json_dir):
    dados = []
    for nome_img in os.listdir(img_dir):
        if not nome_img.endswith(".jpg"):
            continue

        caminho_img = os.path.join(img_dir, nome_img)
        caminho_json = os.path.join(json_dir, nome_img.replace(".jpg", ".json"))

        if not os.path.exists(caminho_img) or not os.path.exists(caminho_json):
            print(f"❌ Arquivo ausente: {nome_img}")
            continue

        try:
            with open(caminho_json, "r") as f:
                info = json.load(f)
            prompt = json_para_string(info)
        except Exception as e:
            print(f"❌ Erro ao ler JSON: {caminho_json} | Erro: {e}")
            continue

        dados.append({
            "image_path": caminho_img,
            "target_text": prompt
        })

    print(f"✅ Total de exemplos carregados: {len(dados)}")
    return Dataset.from_list(dados)

# 4. Transformação para ser aplicada via map()
def transform(batch):
    pixel_values_list = []
    labels_list = []

    for image_path, target_text in zip(batch["image_path"], batch["target_text"]):
        try:
            # Carregar e processar imagem
            image = Image.open(image_path).convert("RGB")
            image = image.resize((640, 480))  # Resolução ajustada
            pixel_values = processor(image, return_tensors="pt").pixel_values[0]

            # Tokenização do texto
            input_ids = processor.tokenizer(
                target_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            ).input_ids[0]

            # Acumular resultados para a lista
            pixel_values_list.append(pixel_values.numpy())
            labels_list.append(input_ids.numpy())

            # Limpeza de recursos de memória
            del image
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Erro ao processar imagem: {image_path}")
            print(f"🧾 Detalhes: {e}")
            pixel_values_list.append(torch.zeros((3, 768, 1024)).numpy())  # Usando zeros
            labels_list.append(torch.full((512,), -100).numpy())  # Preenchendo com -100 (token de pad)

    return {
        "pixel_values": pixel_values_list,
        "labels": labels_list
    }

# 5. Carregar o dataset
dataset = carregar_dados("dataset_receitas/images", "dataset_receitas/annotations")

# 6. Aplica transformação com map (batched=True, tamanho do lote ajustado)
dataset = dataset.map(transform, batched=True, batch_size=1)  # Ajuste no batch_size

# 7. Ajusta o formato para PyTorch com as colunas necessárias (antes de dividir)
dataset.set_format(type="torch", columns=["pixel_values", "labels"])

# 8. Dividir o dataset em treino e validação
train_size = int(0.9 * len(dataset))  # 90% para treino
val_size = len(dataset) - train_size  # 10% para validação
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 9. Argumentos do treinamento
args = Seq2SeqTrainingArguments(
    output_dir="./donut-receitas",
    per_device_train_batch_size=1,
    num_train_epochs=10,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="epoch",  # Avalia ao final de cada época
    dataloader_num_workers=0  # Evitar múltiplos workers para o dataloader
)

# 10. Configura o trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Passando o eval_dataset para a avaliação
)

# 11. Inicia o treino
trainer.train()

print("\n🔒 Salvando o modelo treinado...")

# Salva o modelo treinado
model.save_pretrained("./donut-receitas")

# Salva o tokenizador
processor.tokenizer.save_pretrained("./donut-receitas")

print("✅ Modelo e tokenizador salvos com sucesso!")
