# Importa o PIL para abrir imagens, e o processador + modelo treinado
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Carrega o processor e o modelo treinado anteriormente (na pasta que salvamos)
processor = DonutProcessor.from_pretrained("donut-receitas")
model = VisionEncoderDecoderModel.from_pretrained("donut-receitas")

# Coloca o modelo em modo de avaliação (evita comportamento de treino como dropout)
model.eval()

# Abre uma nova imagem de receita que queremos interpretar
image = Image.open("nova_receita.jpg").convert("RGB")

# Processa a imagem para virar tensor (como o modelo espera)
pixel_values = processor(image, return_tensors="pt").pixel_values

# Define o "prompt", ou seja, o que o modelo deve imaginar que está vendo
# Isso depende do seu JSON de treinamento. Pode ser "<s_receita>" ou algo como "<s>data</s>"
prompt = "<s_receita>"

# Converte o prompt em tokens numéricos (para alimentar o modelo)
decoder_input_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids

# Agora o modelo gera a previsão com base na imagem + prompt
outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=768)

# Decodifica os tokens de volta para texto legível
resultado = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# Exibe o texto final extraído da imagem
print("🧾 Resultado:", resultado)
