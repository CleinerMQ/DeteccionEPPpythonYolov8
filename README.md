# 🦺 Detector de EPP (Equipo de Protección Personal) 🛠️

![Logo](https://img.icons8.com/ios-filled/50/000000/worker-safety-helmet.png)

---

## 🚀 Descripción

Esta aplicación utiliza **YOLO** para la detección en tiempo real de Equipos de Protección Personal (EPP) en video, tales como:

- 🥾 Botas  
- 🧤 Guantes  
- ⛑️ Casco  
- 🧍 Humano  
- 🦺 Chaleco reflectante  

El programa procesa video desde una cámara y muestra las detecciones con cajas de colores y etiquetas, ayudando a garantizar la seguridad en ambientes laborales.

---

## 📦 Tecnologías utilizadas

- Python 🐍  
- OpenCV 🖼️  
- Ultralytics YOLO 🎯  
- Tkinter (interfaz gráfica) 🖥️  
- PIL (Python Imaging Library) 🖼️  
- Threading para procesamiento en segundo plano 🧵  

---

## ⚙️ Dependencias

Para ejecutar este proyecto necesitas instalar las siguientes librerías de Python:

- `ultralytics` (para el modelo YOLO)
- `opencv-python` (procesamiento de video y visión por computadora)
- `pillow` (manipulación de imágenes para la interfaz)
- `tkinter` (interfaz gráfica; generalmente viene preinstalado con Python)
- `numpy` (manejo de arrays y operaciones numéricas)

Puedes instalarlas fácilmente usando:

```bash
pip install ultralytics opencv-python pillow numpy
