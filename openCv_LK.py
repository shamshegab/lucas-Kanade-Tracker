

def get_frames(folder_name):
    PATH = 'C:/Users/omark/Downloads/tracking_data/tracking_data/' + folder_name
    frames = []
    for filename in  sorted(os.listdir(PATH)):
        frame = cv2.imread(PATH +"/"+ filename)
        frames.append(np.array(frame))
    return np.array(frames)

car_frames = get_frames("car")
print(car_frames.shape)

landing_frames = get_frames("landing")
print(landing_frames.shape)




def mark_points_car(frame, p1, marked_car_frames):
    new_frame = np.copy(frame)
    for point in (p1):
        marked_img = cv2.circle(new_frame, (point[0], point[1]), radius= 5, color=(255, 100, 0), thickness = 2 )
    p1 = p1.astype(int)
    w = (p1[0][0]-p1[1][0]) + 40
    h = (p1[3][1]-p1[0][1])*2 +10
    x1 = p1[1][0] -20

    y1 = int(p1[0][1] - 0.4*h)
#     print(x1,y1)
    marked_img = cv2.rectangle(marked_img, (x1, y1), (x1+w, y1+h), (0,255,0), 3)
    marked_car_frames.append(marked_img)
    return marked_car_frames
    
def mark_points_landing(frame, p1, marked_landing_frames):
    new_frame = np.copy(frame)
    for point in (p1):
        marked_img = cv2.circle(new_frame, (point[0], point[1]), radius= 5, color=(255, 100, 0), thickness = 2 )
    p1 = p1.astype(int)
    x1 = p1[0][0] -20
    y1 = p1[0][1] - 20
    x2 = p1[1][0] +20
    y2 = p1[1][1] +20
#     print(x1,y1)
    marked_img = cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0,255,0), 3)
    marked_landing_frames.append(marked_img)
    return marked_landing_frames




new_car_frames = []
p0 = np.float32([[315,167],[145,175], [300,190], [277,248]])
for t in range(car_frames.shape[0]-1):
    p1, st, err = cv2.calcOpticalFlowPyrLK(car_frames[t],car_frames[t+1], p0, None)
    p0 = p1
    new_car_frames = mark_points_car(car_frames[t+1], p1, new_car_frames)




new_landing_frames = []
p0 = np.float32([[448,90],[475,127],[455,107]])
for t in range(landing_frames.shape[0]-1):
    p1, st, err = cv2.calcOpticalFlowPyrLK(landing_frames[t],landing_frames[t+1], p0, None)
    p0 = p1
    new_landing_frames = mark_points_landing(landing_frames[t+1], p1, new_landing_frames)





def make_video(frames, name):
    frames= np.array(frames)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("C:/Users/omark/Downloads/tracking_data/"+name+".mp4",fourcc, 15, (frames.shape[2],frames.shape[1]))
   
    for i in range(len(frames)):
        frame = cv2.resize(frames[i],(frames.shape[2],frames.shape[1]))
        out.write(frame)

    out.release()

make_video(new_car_frames, "cars")
make_video(new_landing_frames, "landing")




