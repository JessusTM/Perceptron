import numpy as np

tamañoEntrada = 100
tamañoCapaOculta = 10
xLinea = np.array([1 if i >= 40 and i < 50 else 0 for i in range(tamañoEntrada)])
xCirculo = np.array(
    [
        1 if (i % 10 - 4.5) * 2 + (i // 10 - 4.5) * 2 <= 8 else 0
        for i in range(tamañoEntrada)
    ]
)

pesosEntradaOculta = np.random.uniform(
    -0.1, 0.1, size=(tamañoEntrada, tamañoCapaOculta)
)
pesosOcultaSalida = np.random.uniform(-0.1, 0.1, size=tamañoCapaOculta)
umbralOculta = np.random.uniform(-0.1, 0.1, size=tamañoCapaOculta)
umbralSalida = 0


def Paso(sumaPonderada):
    return 1 if sumaPonderada > 0 else 0


def Softmax(puntajes):
    expPuntajes = np.exp(puntajes - np.max(puntajes))
    probabilidades = expPuntajes / expPuntajes.sum()
    return probabilidades


def Perceptron(entrada, pesosEntradaOculta, pesosOcultaSalida):
    activacionesOcultas = []

    for j in range(tamañoCapaOculta):
        pesosNeuronas = pesosEntradaOculta[:, j]
        productoPunto = np.dot(entrada, pesosNeuronas)
        sumaPonderada = productoPunto + umbralOculta[j]
        activacion = Paso(sumaPonderada)
        activacionesOcultas.append(activacion)

    sumaPonderadaSalida = np.dot(activacionesOcultas, pesosOcultaSalida) + umbralSalida
    return Softmax([sumaPonderadaSalida, umbralSalida])


def EntrenarPerceptron(entradas, etiquetas, epocas=500, tasaAprendizaje=0.01):
    global pesosEntradaOculta, pesosOcultaSalida
    for _ in range(epocas):
        for entrada, etiqueta in zip(entradas, etiquetas):
            sumaEntradaOculta = np.dot(entrada, pesosEntradaOculta) + umbralOculta
            activacionesOcultas = []
            for x in sumaEntradaOculta:
                activacionesOcultas.append(Paso(x))
            activacionesOcultas = np.array(activacionesOcultas)

            sumaPonderadaSalida = (
                np.dot(activacionesOcultas, pesosOcultaSalida) + umbralSalida
            )
            probabilidadesSalida = Softmax([sumaPonderadaSalida, umbralSalida])

            etiquetaOneHot = np.zeros_like(probabilidadesSalida)
            etiquetaOneHot[etiqueta] = 1

            errorSalida = probabilidadesSalida - etiquetaOneHot
            gradienteSalida = errorSalida

            pesosOcultaSalida -= tasaAprendizaje * np.dot(
                activacionesOcultas, gradienteSalida[0]
            )

            for j in range(tamañoCapaOculta):
                gradienteOculta = gradienteSalida[0] * pesosOcultaSalida[j]
                pesosEntradaOculta[:, j] -= tasaAprendizaje * gradienteOculta * entrada


entradas = np.array([xLinea, xCirculo])
etiquetas = np.array([0, 1])

EntrenarPerceptron(entradas, etiquetas)

salidaLinea = Perceptron(xLinea, pesosEntradaOculta, pesosOcultaSalida)
print(f"Probabilidad de línea   (entrada línea): {salidaLinea[0]}")
print(f"Probabilidad de círculo (entrada línea): {salidaLinea[1]}")

if salidaLinea[0] > salidaLinea[1]:
    print("Es una «Línea»")
else:
    print("Es un «Círculo»")

salidaCirculo = Perceptron(xCirculo, pesosEntradaOculta, pesosOcultaSalida)
print(f"Probabilidad de línea   (entrada círculo): {salidaCirculo[0]}")
print(f"Probabilidad de círculo (entrada círculo): {salidaCirculo[1]}")

if salidaCirculo[0] > salidaCirculo[1]:
    print("Es una «Línea»")
else:
    print("Es un «Círculo»")
