# FaceSecure
This project serves as my thesis, fulfilling the graduation requirement for obtaining a Bachelor's degree in Computer Science. \
Worship activities are essential for religious communities in Indonesia and must be conducted in secure, respected places. However, security threats in these locations jeopardize their safety. To address this, we designed a security system using face recognition technology and deep learning algorithms such as MTCNN and ResNet. Data was collected through interviews and literature studies. The resulting system monitors entrances to help security personnel on their job. The project was made entirely using Python, even the GUI which utilizes PyQt5.

## Features
- **Facial Detection:** The system detects faces within the vicinity of the place of worship using computer vision techniques.
- **Facial Recognition:** Facial data of congregation members are stored in a database. The system compares detected faces with the stored facial data to identify congregation members.
- **Unrecognized Face Handling:** If a detected face is unrecognized, the system captures and stores the image for subsequent review by security personnel.
- **Real-time Alerting:** Unrecognized faces are marked with a red box within the frame in real-time. Additionally, an audible alert is triggered to draw immediate attention to the security breach.
- **Concurrent Processes:** Facial detection, recognition, and alerting processes occur concurrently to ensure timely response to security threats.
- **Suspect Identification:** The system can recognize faces confirmed by authorities as suspects involved in criminal activities and currently wanted. Provided with pictures of suspects from trustworthy sources (e.g., law enforcement), the system alerts security personnel if any suspects are detected.

## How the system works
The system will utilize the external camera to do surveillance and detect any face that is in the frame.\
Upon detection of faces, the system conducts a comparison with facial data stored in the database. In the event of unrecognized faces, the system proceeds to capture and store images within a designated folder for subsequent review. \
Concurrently, the system alerts security personnel by visually highlighting unrecognized faces within the frame using a distinctive red box and emitting an audible alert to prompt immediate attention.

## System Architecture
![Arsitektur Sistem Keamanan drawio](https://github.com/FriendlyMaabuat/FaceSecure/assets/92776515/56f46719-621f-499f-afca-7e637f4e84ee)\
The program has two users, which are the administrator and the security personnel. 

## Users
- Administrator \
Administrators possess comprehensive access to all program functionalities, including but not limited to facial data collection, storage of facial IDs in the database, database manipulation, and model training for the face recognition feature.

- Security Personnel \
Security personnel are granted access exclusively to the live monitoring or surveillance feature and have the ability to review surveillance results.

## User Interface
- Login page \
This page is the initial screen that appears when the system is accessed. Users can enter their username and password in the provided fields and click the login button at the bottom to access the system. \
![image](https://github.com/FriendlyMaabuat/FaceSecure/assets/92776515/ed974de8-9a73-431e-8ae0-f78989a5dbc5)
- Main page for administrators \
This page appears when an administrator successfully logs into the system. It serves as the main dashboard where the administrator can choose to collect facial data and train the model, monitor security through the system, or access reports on the security monitoring results. \
![image](https://github.com/FriendlyMaabuat/FaceSecure/assets/92776515/7f0c46dd-4e1d-4218-af0f-0cef5fef6e95)]
- Main page for security personnel \
This page appears when a security personnel successfully logs into the system. It serves as the main dashboard where the user can choose to monitor security through the system or access reports on the security monitoring results. \
![image](https://github.com/FriendlyMaabuat/FaceSecure/assets/92776515/10783edc-6adc-462f-a9f5-db8d079f3d11)
- Open Cam (Surveillance) \
This page appears when an administrator or security personnel clicks the "Open Cam" button to initiate security monitoring through the system. To activate the camera, the user can click the grey camera button located at the bottom right of the "Open Cam" page. To stop the camera, the user can click the red camera button located at the bottom left of the "Open Cam" page. \
![image](https://github.com/FriendlyMaabuat/FaceSecure/assets/92776515/8bd8f31b-cec4-4f95-98f7-39d2972a1556)
- Load Excel (Surveillance reports/logs) \
This page appears when an administrator or user clicks the "Load Excel" button to access security monitoring reports. On this page, information such as the type of detected objects, similarity percentage, date, time, and ID of the detected objects can be accessed. \
![image](https://github.com/FriendlyMaabuat/FaceSecure/assets/92776515/b0404634-ff32-4a34-a602-88a0a8aaab8f)
- Collect Data (Collecting facial data) \
This page appears when an administrator clicks the "Collect Data" button. It is used for collecting facial data and initiating the model training process. To start data collection, the administrator can click the camera button located at the bottom right of the "Collect Data" page. To stop the facial data collection process, the administrator can click the red cross button at the bottom of the page. To begin the model training process, the administrator can click the AI icon button, the second button from the left at the bottom of the "Collect Data" page. To return to the main dashboard, the administrator can click the back button, the first button from the left at the bottom of the "Collect Data" page. \
![image](https://github.com/FriendlyMaabuat/FaceSecure/assets/92776515/0ff38e90-29a5-4233-abe0-3d24f791697d)

## Examples
- Five faces in the frame: 2 congregation members, 2 unrecognized faces, and 1 suspect. \
![image](https://github.com/FriendlyMaabuat/FaceSecure/assets/92776515/b7155cbc-33e8-418c-a08a-28bf771728e3)
- Four faces in the frame: 2 congregation members, 1 unrecognized face, and 1 suspect. \
![image](https://github.com/FriendlyMaabuat/FaceSecure/assets/92776515/46855295-2e04-43d4-b74a-09d0eb19c10b)
- Three faces in the frame: 1 congregation member, 1 unrecognized face, and 1 suspect. \
![image](https://github.com/FriendlyMaabuat/FaceSecure/assets/92776515/23b3e962-981a-467f-a044-6831a9e908c0)
- Two faces in the frame: 1 congregation member, and 1 unrecognized face. \
![image](https://github.com/FriendlyMaabuat/FaceSecure/assets/92776515/9871c562-71d1-4ab6-b848-8275914b5151)














