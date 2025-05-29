import os
import cv2
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
from threading import Thread
from PIL import Image, ImageTk

# Obtener la ruta actual del ejecutable o script
ruta_base = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(ruta_base, 'best.pt')

# Verificar si el modelo existe
if not os.path.exists(ruta_modelo):
    raise FileNotFoundError(f"Archivo de modelo no encontrado: {ruta_modelo}")

# Cargar el modelo YOLO
modelo = YOLO(ruta_modelo)

# Diccionario para asignar colores a las clases detectadas
colores = {
    'botas': (255, 0, 0),
    'guantes': (0, 255, 0),
    'casco': (0, 0, 255),
    'humano': (255, 255, 0),
    'chaleco': (0, 255, 255)
}

# Variables para control de video
captura_video = None
corriendo = False

# Variable global para el umbral de confianza
umbral_confianza = 0.5

# Función para mejorar el frame
def preprocesar_frame(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    suavizado = cv2.GaussianBlur(gris, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frame_mejorado = clahe.apply(suavizado)
    return cv2.cvtColor(frame_mejorado, cv2.COLOR_GRAY2BGR)

# Función para listar cámaras disponibles
def listar_camaras():
    camaras_disponibles = []
    for i in range(10):  # Probar los primeros 10 índices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camaras_disponibles.append(str(i))
            cap.release()
    return camaras_disponibles

# Función para iniciar la detección de video
def iniciar_video():
    global captura_video, corriendo
    corriendo = True
    seleccion_camara = int(combo_camaras.get())
    captura_video = cv2.VideoCapture(seleccion_camara)

    def procesar_video():
        while corriendo:
            ret, frame = captura_video.read()
            if not ret:
                break
            frame_mejorado = preprocesar_frame(frame)
            resultados = modelo(frame_mejorado)[0]

            for resultado in resultados.boxes:
                clase_id = int(resultado.cls[0])
                confianza = float(resultado.conf[0])
                bbox = resultado.xyxy[0].cpu().numpy()
                nombre_clase = modelo.names[clase_id]

                # Utilizar el umbral de confianza global
                if confianza > umbral_confianza:
                    x1, y1, x2, y2 = bbox.astype(int)
                    color = colores.get(nombre_clase, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    texto = f"{nombre_clase} ({confianza:.2f})"
                    cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Convertir frame para mostrar en Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            etiqueta_video.configure(image=imagen)
            etiqueta_video.image = imagen

        captura_video.release()
        etiqueta_video.image = None

    Thread(target=procesar_video).start()

# Función para detener la detección de video
def detener_video():
    global corriendo
    corriendo = False

# Función para actualizar el umbral de confianza en vivo
def actualizar_umbral(event):
    global umbral_confianza
    umbral_confianza = slider_confianza.get() / 100

# Crear la ventana principal de Tkinter
ventana = tk.Tk()
ventana.title("Detección de Objetos")
ventana.geometry("800x650")
ventana.configure(bg="#2e2e2e")  # Fondo oscuro para un estilo profesional

# Contenedor para controles
frame_controles = tk.Frame(ventana, bg="#2e2e2e")
frame_controles.pack(pady=10)

# ComboBox para seleccionar la cámara
label_camaras = tk.Label(frame_controles, text="Selecciona Cámara:", bg="#2e2e2e", fg="white", font=("Helvetica", 10))
label_camaras.grid(row=0, column=0, padx=5, pady=5)

camaras_disponibles = listar_camaras()
combo_camaras = ttk.Combobox(frame_controles, values=camaras_disponibles, font=("Helvetica", 10))
combo_camaras.current(0)
combo_camaras.grid(row=0, column=1, padx=5, pady=5)

# Slider para ajustar el umbral de confianza
label_confianza = tk.Label(frame_controles, text="Umbral de Confianza:", bg="#2e2e2e", fg="white", font=("Helvetica", 10))
label_confianza.grid(row=1, column=0, padx=5, pady=5)

slider_confianza = tk.Scale(frame_controles, from_=0, to=100, orient=tk.HORIZONTAL, length=200, bg="#4CAF50", fg="white", font=("Helvetica", 10))
slider_confianza.set(50)  # Valor inicial
slider_confianza.grid(row=1, column=1, padx=5, pady=5)
slider_confianza.bind("<Motion>", actualizar_umbral)  # Llama a actualizar_umbral cuando se mueve el slider

# Botón para iniciar la captura de video
boton_iniciar = tk.Button(frame_controles, text="Iniciar Video", command=iniciar_video, bg="#4CAF50", fg="white", font=("Helvetica", 10), width=12)
boton_iniciar.grid(row=2, column=0, padx=5, pady=10)

# Botón para detener la captura de video
boton_parar = tk.Button(frame_controles, text="Parar Video", command=detener_video, bg="#F44336", fg="white", font=("Helvetica", 10), width=12)
boton_parar.grid(row=2, column=1, padx=5, pady=10)

# Marco para mostrar el video (debajo de los controles)
frame_video = tk.Frame(ventana, bg="#1e1e1e", width=320, height=240, highlightbackground="#4CAF50", highlightthickness=2)
frame_video.pack(pady=20)
etiqueta_video = tk.Label(frame_video)
etiqueta_video.pack()

# Hacer la ventana y los controles responsivos
ventana.columnconfigure(0, weight=1)
frame_controles.columnconfigure([0, 1], weight=1)

# Ejecutar la interfaz
ventana.mainloop()
