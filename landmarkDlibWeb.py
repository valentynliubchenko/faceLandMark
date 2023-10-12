import cv2
import dlib

# Создаем объект для работы с камерой
cap = cv2.VideoCapture(0)

# Создаем детектор лиц с использованием dlib
detector = dlib.get_frontal_face_detector()
destination_path = "./model/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(destination_path)

while True:
    # Считываем кадр с камеры
    ret, frame = cap.read()

    if not ret:
        break

    # Преобразуем кадр в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаруживаем лица на кадре
    faces = detector(gray)

    for face in faces:
        # Для каждого обнаруженного лица вычисляем лицевые ключевые точки (landmarks)
        landmarks = predictor(gray, face)

        # Рисуем прямоугольник вокруг лица
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Рисуем лицевые ключевые точки
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # Отображаем результат
    cv2.imshow("Face Landmarks Detection", frame)

    # Для выхода из цикла нажмите клавишу 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы и закрываем окна
cap.release()
cv2.destroyAllWindows()