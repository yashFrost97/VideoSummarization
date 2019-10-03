import cv2
from Tkinter import *
from PIL import ImageTk, Image
import tkMessageBox
import tkFileDialog
import ttk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import glob, os, os.path
from sklearn.cluster import KMeans
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os,subprocess
def generate_frames(path):
    
    out = subprocess.check_output(["ffprobe",path,"-v","0","-select_streams","v","-print_format","flat","-show_entries","stream=r_frame_rate"])
    rate = out.split('=')[1].strip()[1:-1].split('/')
    if len(rate)==1:
        fps = int(rate[0])
    if len(rate)==2:
        fps = int(rate[0])/int(rate[1])
    print fps
    
    vidcap = cv2.VideoCapture(path)
    print path
    success,image = vidcap.read()
    total_count = 0
    success = True
    while success:
        success,image = vidcap.read()
        if total_count%fps==0:
            cv2.imwrite("test%d.jpg" % total_count, image)
            #print(total_count)
        if cv2.waitKey(10) == 27:                     
            break
        total_count += 1
    
      
    #Storing histogram differences as list of list in results
    tkMessageBox.showinfo("Video Summarization","Frames have been generated!")
    counter=0
    index = {}
    images = {}
    results =[]
    while counter<total_count:
        name="test%d.jpg" % counter
        image = cv2.imread(name)
        images[name] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist(image, [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        index[name] = hist
        counter+=fps

    counter=0
    i=0
    for (k, hist) in index.items():
        results.append([])
        name1="test%d.jpg" % counter
        name2="test%d.jpg" % (counter+fps)
        if(counter<total_count-fps):
            d = cv2.compareHist(index[name1], index[name2], cv2.HISTCMP_CORREL)
            results[i].append(d)
            results[i].append(1)
            counter+=fps
            i+=1
    print results  

    #loading results in csv file
    csvData =results
    
    with open('test.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()
    
   
    #Clustering histogram values
    dataset = pd.read_csv('test.csv')
    X = dataset.iloc[:,:].values
    
    wcss = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters = i, init= 'k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,11),wcss)
    #plt.show()
    kmeans = KMeans(n_clusters =3, init= 'k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1], s=100, c='red', label='One')
    plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1], s=100, c='green', label='Two')
    plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1], s=100, c='cyan', label='Three')
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300, c='yellow',label='centroids')
    plt.title("Cluster of frames")
    plt.xlabel("Everything")
    plt.ylabel("Nothing")
    plt.legend()
    plt.show()

    #ndarray to list
    #print y_kmean
    selected = y_kmeans.tolist()
    print selected

    for idx, val in enumerate(selected):
        if val==0:
            selected[idx]=idx
    while 1 in selected:
        selected.remove(1)
    while 2 in selected:
        selected.remove(2)
    print selected
    #creating video from frames
    vid=0
    tkMessageBox.showinfo("Video Summarization","Key moments selected, merging them!")
    maxRange = max(selected) / 5
    print maxRange
    bucket_counter = [0] * (maxRange + 1)

    print len(bucket_counter)
    
    for second in selected:
        temp = second / 5
        bucket_counter[temp] += 1
    file1 = open("MyFile.txt","w") 
    print bucket_counter
    if bucket_counter.count(5)>=15:
        thres = 5
    elif bucket_counter.count(4)>=15:
        thres = 4
    elif bucket_counter.count(3)>=15:
        thres = 3
    elif bucket_counter.count(2)>=15:
        thres = 2
    
    for idx, val in enumerate(bucket_counter):
        if val>=thres:
            clip = str("test%d.mp4" % vid)
            testfile = ffmpeg_extract_subclip(path, (idx-1)*5+1, idx*5, targetname=clip)
            testfile = clip
            print clip
            vid_test = cv2.VideoCapture(testfile)
            if vid_test.isOpened():
                file1.write('file ')
                file1.write(clip)
                file1.write('\n')
            print (idx-1)*5+1, idx*5
            vid +=1
    
    file1.close() 
    filelist = glob.glob(os.path.join("*.jpg"))
    for f in filelist:
        os.remove(f)
    subprocess.call(["ffmpeg", "-f" ,"concat", "-i", "MyFile.txt", "-vcodec", "copy", "yourvideo.mp4"])
    filelist = glob.glob(os.path.join("test*.mp4"))

    
    tkMessageBox.showinfo("Video Summarization","Your video has been summarized!")
def exitapp():
    tkMessageBox.showinfo("Video Summarization","Exiting the app...")
    answer = tkMessageBox.askquestion("Video Summarization","Do you really want to exit?")
    if answer == 'yes':
        root.destroy()
    
def browse_button():
    file = tkFileDialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("mp4 files","*.mp4"),("all files","*.*")))
    generate_frames(file)

def about_app():
    
    
    quote = """The main aim of Video summarization is to provide clear analysis of video by
removing duplications and extracting key frames from the video. Massive
growth in video content poses problem of information overload and
management of content. In order to manage the growing videos on the web and
also to extract an efficient and valid information from the videos, more attention
has to be paid towards video and image processing technologies. Video
summarization is a mechanism to produce a short summary of a video to give to
the user a synthetic and useful visual abstract of video sequence; it can either be
an image (key frames) or moving images. Video summarization is a vital
process that facilitates well-organized storage, quick browsing, and retrieval of
large collection of video data without losing important aspects. In terms of
browsing and navigation, a good video abstract will enable the user to get
maximum information about the target video sequence in a specified time
limitation or adequate information in the minimum time."""
    label = Label(root, text=quote, pady=10,padx=0, justify=LEFT)
    label.config(font=("Arial",20))
    label.pack(side=TOP)
    
    label2 = Label(root,pady=30, text="Developed By\n AKHIL RAUT\n PRASHANT PORWAL\n YASH PATANGE")
    label2.config(font=("Arial",30))
    label2.pack(side=TOP)
    root.mainloop()
    


#desktop application
root = Tk()

root.title("Video Summarization")

myMenu = Menu(root,bg="blue")
myMenu.add_cascade(label="Select File",command=browse_button)
myMenu.add_cascade(label="About",command=about_app)
myMenu.add_cascade(label="Exit",command=exitapp)

# canvas goes here

root.configure(menu=myMenu)

canvas = Canvas(width=1280, height = 700, bg='gray')
canvas.pack(expand=YES, fill=BOTH)
img = Image.open('project_home.png')
canvas.image = ImageTk.PhotoImage(img)
canvas.create_image(0, 0, image = canvas.image, anchor='nw')

root.mainloop()
