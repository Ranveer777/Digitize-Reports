# Digitize Reports
A web-app to digitize medical reports & provide the data in various usable formats, along with some key features for user convenience.

## End Users
The application is built for insurance companies, who struggle with printed lab reports.
Digitization of these reports enables faster verification and provides other functionalities, which help in deciding insurance packages for the client.

## How does the app work?
1. Text Detection in separate columns using [OpenCV-Python](https://pypi.org/project/opencv-python/).
2. Text Recognition using [PyTesseract](https://pypi.org/project/pytesseract/).
3. Processing the data & displaying the report in a tabular format. Along with 'HIGH' / 'LOW' indication.
4. Storage of reports in a MongoDB database using [PyMongo](https://pypi.org/project/pymongo/).
5. Using the [Pandas](https://pypi.org/project/pandas/) library to produce a trend chart based on previous reports.

## Future Updates:
1. Verification of MediClaim by comparison with Health Check Parameters.
2. NLP based model for better accuracy & compatibility with varying structures of reports.
3. Insurance Coverage suggestions by health prediction using previous reports.
