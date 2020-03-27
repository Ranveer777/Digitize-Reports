from flask import Flask, render_template, request, redirect, send_from_directory, session
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import math
from scipy import ndimage
import pytesseract
from pdf2image import convert_from_path
import json
import csv
import datetime
import glob
import re

# trend chart stuff
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd

# mongoDB stuff
from pymongo import MongoClient

cluster = MongoClient("mongodb+srv://dbVikrant:docomo3g@cluster0-giotq.mongodb.net/test?retryWrites=true&w=majority")

db = cluster['test']
collection = db['test']

app = Flask(__name__)

# Session key:
app.secret_key = 'psj1612'

app.config['image_upload'] = 'static/user_uploads'
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "PDF"]

# Orientation correction & adjustment


def orientation_correction(img, save_image=False):
    # GrayScale Conversion for the Canny Algorithm
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Canny Algorithm for edge detection was developed by John F. Canny not Kennedy!! :)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    # Using Houghlines to detect lines
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    # Finding angle of lines in polar coordinates
    angles = []
    for x1, y1, x2, y2 in lines[0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Getting the median angle
    median_angle = np.median(angles)

    # Rotating the image with this median angle
    img_rotated = ndimage.rotate(img, median_angle)

    if save_image:
        cv2.imwrite('orientation_corrected.jpg', img_rotated)
    return img_rotated


def result_coords(im):
    string = ''
    hocr = pytesseract.image_to_pdf_or_hocr(im, extension='hocr')
    with open('tanayhocr.txt', "w+b") as f:
        f.write(bytearray(hocr))

    with open('tanayhocr.txt', 'r') as f:
        string += f.read()

    keywords = ['Result', 'RESULT', 'Value', 'VALUE']
    for word in keywords:
        result = 0
        result = string.find('>' + word, 0, len(string))
        # print(word, result)
        if result == -1:
            continue
        break
    i = 0
    for x in range(result, 0, -1):
        if string[x] == 'x':
            i += 1
            if i == 2:
                break

    ans = string[x + 2: result].split(';')[0]
    ans = ans.split(' ')
    return int(ans[0]), int(ans[1]), int(ans[2]), int(ans[3])


def column_3(image_file, search, filename):
    im = image_file
    # initializing the list for storing the coordinates
    coordinates1 = []
    coordinates2 = []
    coordinates3 = []

    input_img = cv2.imread(im)  # image read
    img_rotated = orientation_correction(input_img)

    # load the image, clone it, and setup the mouse callback function
    image = img_rotated

    dimensions = image.shape
    height = image.shape[0]
    width = image.shape[1]
    x1, y1, x2, y2 = result_coords(image)

    coordinates1 = [(0, y2), (x1 - 10, int(0.7722 * height))]
    coordinates2 = [(x1 - 10, y2), (x2 + 30, int(0.7722 * height))]
    coordinates3 = [(x2 + 30, y2), (int(width), int(0.7722 * height))]

    # coordinates1 = [(0, y2), (x1 - 10, int(height))]
    # coordinates2 = [(x1 - 10, y2), (x2 + 30, int(height))]
    # coordinates3 = [(x2 + 30, y2), (int(width), int(height))]

    cv2.rectangle(image, coordinates1[0], coordinates1[1], (0, 0, 255), 2)
    cv2.rectangle(image, coordinates2[0], coordinates2[1], (0, 0, 255), 2)
    cv2.rectangle(image, coordinates3[0], coordinates3[1], (0, 0, 255), 2)

    image_roi1 = image[coordinates1[0][1]:coordinates1[1][1], coordinates1[0][0]:coordinates1[1][0]]
    image_roi2 = image[coordinates2[0][1]:coordinates2[1][1], coordinates2[0][0]:coordinates2[1][0]]
    image_roi3 = image[coordinates3[0][1]:coordinates3[1][1], coordinates3[0][0]:coordinates3[1][0]]

    text1 = pytesseract.image_to_string(image_roi1, lang='eng')
    text2 = pytesseract.image_to_string(image_roi2, lang='eng')
    text3 = pytesseract.image_to_string(image_roi3, lang='eng')

    hp = list(text1.split('\n'))
    hp = [x.strip('.') for x in hp if x != ',' and x != ' ' and x != '' and x != '.']

    res = list(text2.split('\n'))
    res = [x for x in res if x != ',' and x != ' ' and x != '' and x != '.']

    ran = list(text3.split('\n'))
    ran = [x for x in ran if x != ',' and x != ' ' and x != '' and x != '.']

    # convert ranges to a usable tuple of floats : (low, high)
    float_range = []
    for item in ran:
        # val = [float(s) for s in item.split() if s.isdigit()]
        val = re.findall(r"[-+]?\d*\.\d+|\d+", item)
        new = (float(val[0]), float(val[1]))
        float_range.append(new)

    # for result values w/o comma
    new_res = []
    for item in res:
        if ',' in item:
            new_res.append(item.replace(',', ''))
        else:
            new_res.append(item)

    # compare result with range
    comment = []
    for item in zip(new_res, float_range):
        if float(item[0]) > (item[1][0] + 0.6 * (item[1][1] - item[1][0])):
            comment.append('HIGH')
        # elif float(item[0]) < (item[1][0] + 0.4*(item[1][1] - item[1][0])/2):
            # print(item[0], '> LOW <', item[1])
        else:
            comment.append(' ')
    headings = ['differential count', 'investigation']
    report = dict()
    li = list(zip(res, ran, comment))

    i = 0
    for name in hp:
        if i == len(li):
            break
        if name.lower() in headings:
            li.insert(i, (' ', ' ', ' '))
        report[name] = li[i]
        i += 1

    global email_id
    email_id = find_email(image)
    fin_rep = dict()
    now = str(datetime.datetime.now())
    fin_rep['_id'] = search + '_' + now
    fin_rep['Data'] = report
    fin_rep['Email-id'] = email_id

    json_file = 'static/jsons/' + filename.rsplit(".", 1)[0] + '.json'
    with open(json_file, 'w') as j:
        json.dump(fin_rep, j)

    return hp, li


def column_4(image_file, search, filename):
    coordinates1 = []
    coordinates2 = []
    coordinates3 = []
    coordinates4 = []
    columns4 = []
    im = image_file
    input_img = cv2.imread(im)  # image read
    img_rotated = input_img
    image = img_rotated

    dimensions = image.shape
    height = image.shape[0]
    width = image.shape[1]

    coordinates1 = [(0, int(0.335 * height)), (int(0.310 * width), int(0.6850 * height))]
    coordinates2 = [(int(0.320 * width), int(0.335 * height)), (int(0.472 * width), int(0.6850 * height))]
    coordinates3 = [(int(0.472 * width), int(0.335 * height)), (int(0.700 * width), int(0.6850 * height))]
    coordinates4 = [(int(0.700 * width), int(0.335 * height)), (width, int(0.6850 * height))]
    columns4 = [coordinates1, coordinates2, coordinates3, coordinates4]

    cv2.rectangle(image, coordinates1[0], coordinates1[1], (0, 0, 255), 2)
    cv2.rectangle(image, coordinates2[0], coordinates2[1], (0, 0, 255), 2)
    cv2.rectangle(image, coordinates3[0], coordinates3[1], (0, 0, 255), 2)
    cv2.rectangle(image, coordinates4[0], coordinates4[1], (0, 0, 255), 2)

    rep = []
    for coordinates in columns4:
        image_roi = image[coordinates[0][1]:coordinates[1][1], coordinates[0][0]:coordinates[1][0]]
        text = pytesseract.image_to_string(image_roi)
        rep.append(list(text.split('\n')))

    hp = rep[0]
    hp = [x.replace('.', ' ') for x in hp if x != ',' and x != ' ' and x != '' and x != '.']

    res = rep[1]
    res = [x for x in res if x != ',' and x != ' ' and x != '' and x != '.']

    ran = rep[2]
    ran = [x for x in ran if x != ',' and x != ' ' and x != '' and x != '.']

    method = rep[3]
    method = [x for x in method if x != ',' and x != ' ' and x != '' and x != '.']

    float_range = []
    for item in ran:
        # val = [float(s) for s in item.split() if s.isdigit()]
        val = re.findall(r"[-+]?\d*\.\d+|\d+", item)
        new = (abs(float(val[0])), abs(float(val[1])))
        float_range.append(new)

    # for result values w/o comma
    new_res = []
    for item in res:
        if ',' in item:
            new_res.append(item.replace(',', ''))
        else:
            new_res.append(item)

    # compare result with range
    comment = []
    for item in zip(new_res, float_range):
        if float(item[0]) > (item[1][0] + 0.6*(item[1][1] - item[1][0])):
            comment.append('HIGH')
        # elif float(item[0]) < (item[1][0] + 0.4*(item[1][1] - item[1][0])/2):
        #     print(item[0], ' ', item[1])
        else:
            comment.append(' ')
    headings = ['differential count', 'investigation']
    report = dict()
    li = list(zip(res, ran, method, comment))

    i = 0
    for name in hp:
        if i == len(li):
            break
        if name.lower() in headings:
            li.insert(i, (' ', ' ', ' ', ' '))
        report[name] = li[i]
        i += 1

    global email_id
    email_id = find_email(image)
    fin_rep = dict()
    now = str(datetime.datetime.now())
    fin_rep['_id'] = search + '_' + now
    fin_rep['Data'] = report
    fin_rep['Email-id'] = email_id

    json_file = 'static/jsons/' + filename.rsplit(".", 1)[0] + '.json'
    with open(json_file, 'w') as j:
        json.dump(fin_rep, j)

    return hp, li


def allowed_image(filename):
    # We only want files with a . in the filename
    if "." not in filename:
        return False

    # Split the extension from the filename
    ext = filename.rsplit(".", 1)[1]

    # Check if the extension is in ALLOWED_IMAGE_EXTENSIONS
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


def find_email(im):
    hocr = pytesseract.image_to_pdf_or_hocr(im, extension='hocr')
    with open('testhocr.txt', "w+b") as f:
        f.write(bytearray(hocr))

    string = ''
    with open('testhocr.txt', 'r') as f:
        string += f.read()

    index = 0
    while True:
        if string.find('@', index, len(string)) != None:
            result = string.find('@', index, len(string))
            m = 0
            for i in range(result, 0, -1):
                if string[i] == '>':
                    break
                m = i

            n = 0
            for j in range(result, len(string)):
                if string[j] == '<':
                    break
                n = j

            hey = ''
            for x in range(m, n + 1):
                hey += string[x]

            if len(hey) < 5 or len(hey) > 40:
                index = result + 1
                if index >= len(string):
                    break
                continue

            break

        elif string.find('.com', index, len(string)) != None:
            result = string.find('.com', index, len(string))
            m = 0
            for i in range(result, 0, -1):
                if string[i] == '>':
                    break
                m = i

            n = 0
            for j in range(result, len(string)):
                if string[j] == '<':
                    break
                n = j

            hey = ''
            for x in range(m + 1, n):
                hey += string[x]

            if len(hey) < 7 or len(hey) > 40:
                index = result + 1
                if index >= len(string):
                    break
                continue

            break

        else:
            print('Not found')
            break

    bad = [';', '&', ',', ':', '/', '|', ']', '[', '}', '{', ')', '(']
    for item in bad:
        if item in hey:
            hey = hey.split(item)[1]
            break

    for item in bad:
        if item in hey:
            hey = hey.split(item)[0]
            break

    return hey


@app.route('/', methods=['GET', 'POST'])
def new_home():
    search = ''
    if request.method == 'POST':
        first = request.form['first']
        mobile = request.form['mobile']
        search = first + mobile
        session['newSearch'] = search
        all = collection.find()
        exists = 0
        for item in all:
            if search in item['_id']:
                exists = 1
        return render_template('newhome.html', exists=exists, search=search)
    return render_template('newhome.html')


@app.route('/homepage', methods=['GET', 'POST'])
def upload_image():
    if 'newSearch' in session:
        search = session['newSearch']
    else:
        search = ''

    if request.method == 'POST':
        if request.files:
            image = request.files['image']

            if image.filename == "":
                print("No filename")
                return redirect(request.url)

            if allowed_image(image.filename):
                err_msg = 1
                session['newErr_Msg'] = err_msg
                filename = secure_filename(image.filename)
                session['newFilename'] = filename
                image.save(os.path.join(app.config['image_upload'], filename))
                return render_template('homepage.html', filename=filename, err_msg=err_msg, search=search)

            else:
                err_msg = -1
                session['newErr_Msg'] = err_msg
                print("This extension is not allowed!")
                return render_template('homepage.html', err_msg=err_msg, search=search)

    return render_template('homepage.html', search=search)


@app.route('/table', methods=['GET', 'POST'])
def columns():
    if request.method == 'POST':
        num_col = request.form['column_model']
        err_msg = session['newErr_Msg']
        filename = session['newFilename']
        search = session['newSearch']
        
        session['newNum_Col'] = num_col

        ext = filename.rsplit(".", 1)[1]
        img_file = 'static/user_uploads/' + filename
        if ext.upper() == 'PDF':
            pages = convert_from_path(img_file)
            img_name1 = img_file.strip(ext)

            img_name2 = img_name1 + 'jpg'
            pages[0].save(img_name2, 'JPEG')
            os.remove(img_file)
            img_file = img_name2

        email_id = ''
        if num_col == '3':
            hp, li = column_3(img_file, search, filename)
            # email_id = find_email(img_file)
            return render_template('table.html', err_msg=err_msg, report=zip(hp, li), col=3, email_id=email_id)
        elif num_col == '4':
            hp, li = column_4(img_file, search, filename)
            email_id = find_email(img_file)
            return render_template('table.html', err_msg=err_msg, report=zip(hp, li), col=4, email_id=email_id)
    return redirect(request.url)


@app.route('/table-uploaded', methods=['GET', 'POST'])
def table_uploaded():
    search = session['newSearch']
    filename = session['newFilename']

    json_file = 'static/jsons/' + filename.rsplit(".", 1)[0] + '.json'
    img_file = 'static/user_uploads/' + filename.rsplit(".", 1)[0] + '.' + filename.rsplit(".", 1)[1]
    if 'verify' in request.form:
        if request.form['verify'] == 'Continue':
            with open(json_file) as j:
                store = json.load(j)
            collection.insert_one(store)
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from email.mime.base import MIMEBase
            from email import encoders

            fromaddr = "priths.jaunjale@gmail.com"
            toaddr = "priths.jaunjale@gmail.com"

            # instance of MIMEMultipart
            msg = MIMEMultipart()

            # storing the senders email address
            msg['From'] = fromaddr

            # storing the receivers email address
            msg['To'] = toaddr

            # storing the subject
            msg['Subject'] = "Subject of the Mail"

            body = ''
            with open('static/text_files/verification.txt', 'r') as v:
                body += v.read()
            # string to store the body of the mail

            # attach the body with the msg instance
            msg.attach(MIMEText(body, 'plain'))

            # open the file to be sent
            filenm = "Lab Report"
            attachment = open(img_file, "rb")

            # instance of MIMEBase and named as p
            p = MIMEBase('application', 'octet-stream')

            # To change the payload into encoded form
            p.set_payload((attachment).read())

            # encode into base64
            encoders.encode_base64(p)

            p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

            # attach the instance 'p' to instance 'msg'
            msg.attach(p)

            # creates SMTP session
            s = smtplib.SMTP('smtp.gmail.com', 587)

            # start TLS for security
            s.starttls()

            # Authentication
            s.login(fromaddr, "googlepw@160521120510")

            # Converts the Multipart msg into a string
            text = msg.as_string()

            # sending the mail
            s.sendmail(fromaddr, toaddr, text)

            # terminating the session
            s.quit()
            return render_template('homepage.html', search=search)
        elif request.form['verify'] == 'Cancel':
            if os.path.exists(json_file):
                os.remove(json_file)
            if os.path.exists(img_file):
                os.remove(img_file)
            return render_template('homepage.html', search=search)

    if 'database' in request.form:
        if os.path.exists(json_file):
            os.remove(json_file)
        if os.path.exists(img_file):
            os.remove(img_file)
        return render_template('homepage.html', search=search)
    return render_template('newhome.html')


@app.route('/view_uploaded', methods=['GET', 'POST'])
def view_docs():
    search = session['newSearch']

    all = collection.find()
    indi = []
    for item in all:
        if search in item['_id']:
            indi.append(item)
    return render_template('viewdocs.html', all=indi, search=search)


@app.route('/downloaded', methods=['GET', 'POST'])
def downloaded():
    search = session['newSearch']

    # json_file = 'static/jsons/' + filename.rsplit(".", 1)[0] + '.json'

    for key in request.form:
        if key.startswith('download_'):
            mongoid = key.partition('_')[-1]
            value = request.form[key]
    if 'csv' in value:
        file = collection.find_one({'_id':mongoid})
        csv_file = file['_id'] + '.csv'
        with open('static/csvs/' + csv_file, 'w') as c:
            csvwriter = csv.writer(c)

            hdr = ['Health Parameter', 'Result', 'Range']
            csvwriter.writerow(hdr)

            for key, value in file['Data'].items():
                csvwriter.writerow([key, value[0], value[1]])

        return send_from_directory(directory='static/csvs', filename=csv_file)

    if 'json' in value:
        file = collection.find_one({'_id':mongoid})
        json_file = file['_id'] + '.json'
        with open('static/jsons/' + json_file, 'w') as j:
            json.dump(file, j, indent=2)

        return send_from_directory(directory='static/jsons', filename=json_file)
    return render_template('viewdocs.html')


@app.route('/trend-chart', methods=['GET', 'POST'])
def trend_chart():
    color = ['olive', 'skyblue', 'green', 'yellow', 'red', 'cyan', 'blue', 'magenta', 'black']
    chart_done = 0
    if request.form.get('health'):
        params = request.form.getlist('health')
        # all_rep = collection.find()
        all_json = sorted(glob.glob("static/sample_jsons/*.json"))
        compact = []
        for item in all_json:
            with open(item) as d:
                data = json.load(d)
            compact.append(data)

        new = dict()
        new['x'] = range(1, 6)
        li = []
        for par in params:
            for data in compact:
                for key, value in data.items():
                    if key == par:
                        value[0] = value[0].replace(',', '')
                        li.append(float(value[0]))
                        break
            new_li = list(li)
            new[par] = new_li
            li.clear()

        print(new)
        df = pd.DataFrame(new)
        for item in params:
            plt.plot('x', item, data=df, marker='o')
        plt.grid()
        plt.legend()
        plt.savefig('static/trend/hello.png', bbox_inches='tight')
        plt.close()
        chart_done = 1
        return send_from_directory(directory='static/trend', filename='hello.png')
    return render_template('trendchart.html', chart_done=chart_done)


if __name__ == '__main__':
    glob_filename = ''
    app.run(debug=True)
