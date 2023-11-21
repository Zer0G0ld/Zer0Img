import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# CARREGA O MODELO PRÉ-TREINADO PARA GERAÇÃO DE IMAGENS DO TENSORFLOW
model = hub.load("https://tfhub.dev/google/magenta/arbitray-image-stylization-v1-256/2")

def generate_image(prompt):
    # CARREGA O MODELO DE PROMPT FONECIDA PELO ÚSUARIO
    image = tf.image.decode_image(tf.io.read_file(prompt), channel=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # REDIMENSIONA A IMAGEM PARA O TAMANHO ESPERADO PELO MODELO
    image = tf.image.resize(image, [256, 256])
    
    # GERA A IMAGEM DE SAÍDA BASEADA NO PROMPT FORNECIDO
    stlized_image = model(tf.convert_to_tensor([image]))
    
    # CONVERTE A IMAGEM DE SAÍDA BASEADA NO PROMPT FORNECIDO
    stylized_image = tf.image.convert_image_dtype(stylized_image[0], dtype=tf.uint8)
    stylized_image = tf.squeeze(stylized_image)
    
    # SALVA A IMAGEM GERADA EM DISCO
    tf.io.write_file('output.jpg', tf.image.encode_jpeg(stylized_image))
    
    return stylized_image

# USOO
user_prompt = 'input.jpg' # CAMINHO DA IMAGEM
generated_image = generate_image(user_prompt)
print("Imagem gerada com sucesso!")
