from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from PIL import Image
import torch
import json
import os
import gc

# 1. Carrega o modelo e o processador Donut
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# 2. Converte o JSON em string formatada
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

# 3. Carrega e valida os dados do diretório
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

# 4. Função de pré-processamento sob demanda (set_transform)
def transform(example):
    try:
        image = Image.open(example["image_path"]).convert("RGB")
        image = image.resize((1280, 960))  # Reduz a resolução da imagem
        pixel_values = processor(image, return_tensors="pt").pixel_values[0]

        input_ids = processor.tokenizer(
            example["target_text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).input_ids[0]

        del image
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "pixel_values": pixel_values,
            "labels": input_ids
        }

    except Exception as e:
        print(f"❌ Erro ao processar {example['image_path']}: {e}")
        return None  # Importante: Trainer ignora amostras inválidas

# 5. Carrega dataset e aplica transformação sob demanda
dataset = carregar_dados("dataset_receitas/images", "dataset_receitas/annotations")
dataset.set_transform(transform)

# 6. Argumentos de treino
args = Seq2SeqTrainingArguments(
    output_dir="./donut-receitas",
    per_device_train_batch_size=1,
    num_train_epochs=10,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available()  # Ativa fp16 apenas se houver GPU
)

# 7. Treinador
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=dataset
)

# 8. Início do treino
trainer.train()

