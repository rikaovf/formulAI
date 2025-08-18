from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

# Carregar o modelo treinado e o processador
processor = DonutProcessor.from_pretrained("./donut-receitas")
model = VisionEncoderDecoderModel.from_pretrained("./donut-receitas")

# Função de predição
def predizer_prescricao(imagem_path):
    # Carregar e pré-processar a imagem
    image = Image.open(imagem_path).convert("RGB")
    image = image.resize((640, 480))  # Ajuste de tamanho conforme o treinamento
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Gerar a predição (com decodificação)
    output = model.generate(pixel_values, max_new_tokens=500)

    # Converter a saída (IDs) para texto
    predicao = processor.decode(output[0], skip_special_tokens=True)
    
    return predicao

# Testando a predição com uma nova imagem de receita
imagem_nova = "teste.jpg"
texto_predito = predizer_prescricao(imagem_nova)

print(f"Texto Predito: {texto_predito}")
