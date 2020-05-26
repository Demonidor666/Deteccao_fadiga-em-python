# Caso execute no terminal linux passar os argumentos na linha
# python deteccao_sonolencia.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python deteccao_sonolencia.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

# toca o alarme
def sound_alarm(path):
    playsound.playsound(path)



def eye_aspect_ratio(eye):
    # calcular as distâncias euclidianas entre os dois conjuntos.
    # vertical eye landmarks (x, y) coordenadas
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # calcular as distâncias euclidianas entre os dois conjuntos.
    # horizontal eye landmark (x, y) coordenadas
    C = dist.euclidean(eye[0], eye[3])

    # calcular a proporção do olho
    ear = (A + B) / (2.0 * C)

    # retorno do calculo das distancias.
    return ear


# habilitar essa parte do codigo para a execução passado os argumentos
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,
#                help="arquivo landmarks_face")
#ap.add_argument("-a", "--alarm", type=str, default="",
#                help="arquivo do alarme")
#ap.add_argument("-w", "--webcam", type=int, default=0,
#                help="indice da camera incorporada no pc")
#args = vars(ap.parse_args())

# defina duas constantes, uma para a proporção do olho para indicar
# piscar EYE_AR_THRESH, em seguida, uma segunda constante EYE_AR_CONSEC_FRAMES para o número consecutivos
# de quadros nos quais o olho permanece fechado.
EYE_AR_THRESH = 0.28 # 0.3
EYE_AR_CONSEC_FRAMES = 46 # 48

# A constante COUNTER vai contar os quadros.
#  Seto a constante ALARM_ON para false
COUNTER = 0
ALARM_ON = False

# inicializando a parte do reconhecimento da face pelo Dlib
print("[INFO] Carregando shape_predictor_68_face_landmarks.dat...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("recursos\shape_predictor_68_face_landmarks.dat")

# pegue os índices dos pontos de referência faciais à esquerda e
# olho direito, respectivamente
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Startando a camera do noot
print("[INFO] Começando a trasmissão do video...")
#vs = VideoStream(src=args["webcam"]).start()
vs = VideoStream(src=0).start()
time.sleep(1.0)

# loop sobre os frames do fluxo de vídeo
while True:
    # lendo o fluxo  dos quadros e conventendo na scala de cinza
    frame = vs.read()
    frame = imutils.resize(frame, width=550)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detenctando um rosto na escala de cinza.
    rects = detector(gray, 0)

    # laço for varrendo os rosto detectados no rects
    for rect in rects:
        # determina os pontos de referência faciais da região do rosto e, em seguida,
        # converte o marco facial (x, y) -coordenada em um NumPy Array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extrai as coordenadas do olho esquerdo e direito e usa as
        # coordenadas para calcular a proporção dos olhos dos dois olhos
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # calcula a média da proporção dos olhos dos dois olhos
        ear = (leftEAR + rightEAR) / 2.0

        # calcula o convex hull para o olho esquerdo e direito e, em seguida,
        # desenha cada um dos olhos
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # verifica se a proporção do olho está abaixo do piscar
        # threshold e, nesse caso, aumenta o contador de quadros.
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # se o numero de quadros for maior ou igual o valor da constante que definimos acima
            # se for verdadeiro sera emitdo o nosso alarme nesse caso o arquivo de som.
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # se o alarme não estiver ativado, sera setado true
                if not ALARM_ON:
                    ALARM_ON = True

                    # para o ativar o alarme colocamos uma Thread para executar em segundo plano
                    # e passamos para o metodo sound_alarm o arquivo de som
                    if ALARM_ON:
                        t = Thread(target=sound_alarm("recursos/alarm.wav"))
                        t.deamon = True
                        t.start()

                # Quando o alarme for emitido e printado na tela que foi detectado sonolencia.
                cv2.putText(frame, "SONOLENCIA DETECTADO!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # caso contrário, se proporção dos olhos estiver abaixo dos quadros
        # então redefina o contador para zeros e o alarme para false
        else:
            COUNTER = 0
            ALARM_ON = False

        # desenha a proporção dos olhos computados no quadro para ajudar
        # com depuração e configuração da proporção correta dos olhos
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # mostra a janela com o video
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # se a tecla `q` foi pressionada, interrompe o loop
    if key == ord("q"):
        print("[INFO] Fim da Trasmissão")
        break

# limpa o buffer de memoria.
cv2.destroyAllWindows()
vs.stop()

