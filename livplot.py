import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random
import os

fig = plt.figure(figsize=(6,8))
ax1 = fig.add_subplot(2,2,1, title='Male')
ax2 = fig.add_subplot(2,2,2, title='Female')
ax3 = fig.add_subplot(2,2,3, title='Below_18')
ax4 = fig.add_subplot(2,2,4, title='Below_30')
#ax5 = fig.add_subplot(2,2,4, title='Above_30')

def animate(i):
    try:
        pullData = open("emotion_recorded.txt","r").read()
        dataArray = pullData.split('\n')
        male_sad = []
        male_angry = []
        male_disgust=[]
        male_scared=[]
        male_happy=[]
        male_surprised=[]
        male_neutral=[]
        male_yar=[]

        female_sad = []
        female_angry = []
        female_disgust=[]
        female_scared=[]
        female_happy=[]
        female_surprised=[]
        female_neutral=[]
        female_yar=[]

        below_18_sad = []
        below_18_angry = []
        below_18_disgust=[]
        below_18_scared=[]
        below_18_happy=[]
        below_18_surprised=[]
        below_18_neutral=[]
        below_18_yar=[]

        below_30_sad = []
        below_30_angry = []
        below_30_disgust=[]
        below_30_scared=[]
        below_30_happy=[]
        below_30_surprised=[]
        below_30_neutral=[]
        below_30_yar=[]

        above_30_sad = []
        above_30_angry = []
        above_30_disgust=[]
        above_30_scared=[]
        above_30_happy=[]
        above_30_surprised=[]
        above_30_neutral=[]
        above_30_yar=[]

        for eachLine in dataArray:
            if len(eachLine)>1:
                k=eachLine.split(',')

                if k[8] == 'Male':
                    male_angry.append(float(k[0]))
                    male_disgust.append(float(k[1]))
                    male_scared.append(float(k[2]))
                    male_happy.append(float(k[3]))
                    male_sad.append(float(k[4]))
                    male_surprised.append(float(k[5]))
                    male_neutral.append(float(k[6]))
                    male_yar.append(float(k[7]))
                elif k[8] == 'Female':
                    female_angry.append(float(k[0]))
                    female_disgust.append(float(k[1]))
                    female_scared.append(float(k[2]))
                    female_happy.append(float(k[3]))
                    female_sad.append(float(k[4]))
                    female_surprised.append(float(k[5]))
                    female_neutral.append(float(k[6]))
                    female_yar.append(float(k[7]))

                if int(k[9]) <= 18:
                    below_18_angry.append(float(k[0]))
                    below_18_disgust.append(float(k[1]))
                    below_18_scared.append(float(k[2]))
                    below_18_happy.append(float(k[3]))
                    below_18_sad.append(float(k[4]))
                    below_18_surprised.append(float(k[5]))
                    below_18_neutral.append(float(k[6]))
                    below_18_yar.append(float(k[7]))
                elif int(k[9]) <= 40:
                    below_30_angry.append(float(k[0]))
                    below_30_disgust.append(float(k[1]))
                    below_30_scared.append(float(k[2]))
                    below_30_happy.append(float(k[3]))
                    below_30_sad.append(float(k[4]))
                    below_30_surprised.append(float(k[5]))
                    below_30_neutral.append(float(k[6]))
                    below_30_yar.append(float(k[7]))
                else:
                    above_30_angry.append(float(k[0]))
                    above_30_disgust.append(float(k[1]))
                    above_30_scared.append(float(k[2]))
                    above_30_happy.append(float(k[3]))
                    above_30_sad.append(float(k[4]))
                    above_30_surprised.append(float(k[5]))
                    above_30_neutral.append(float(k[6]))
                    above_30_yar.append(float(k[7]))

        ax1.clear()
        ax1.plot(male_yar,male_sad, label='Neutral')
        ax1.plot(male_yar,male_angry, label='Angry')
        ax1.plot(male_yar,male_disgust, label='Disgust')
        ax1.plot(male_yar,male_scared, label='Scared')
        ax1.plot(male_yar,male_happy, label='Happy')
        ax1.plot(male_yar,male_surprised, label='Surprise')
        ax1.plot(male_yar,male_neutral, label='Sad')

        ax2.clear()
        ax2.plot(female_yar,female_sad)
        ax2.plot(female_yar,female_angry)
        ax2.plot(female_yar,female_disgust)
        ax2.plot(female_yar,female_scared)
        ax2.plot(female_yar,female_happy)
        ax2.plot(female_yar,female_surprised)
        ax2.plot(female_yar,female_neutral)

        ax3.clear()
        ax3.plot(below_18_yar,below_18_sad)
        ax3.plot(below_18_yar,below_18_angry)
        ax3.plot(below_18_yar,below_18_disgust)
        ax3.plot(below_18_yar,below_18_scared)
        ax3.plot(below_18_yar,below_18_happy)
        ax3.plot(below_18_yar,below_18_surprised)
        ax3.plot(below_18_yar,below_18_neutral)

        ax4.clear()
        ax4.plot(below_30_yar,below_30_sad)
        ax4.plot(below_30_yar,below_30_angry)
        ax4.plot(below_30_yar,below_30_disgust)
        ax4.plot(below_30_yar,below_30_scared)
        ax4.plot(below_30_yar,below_30_happy)
        ax4.plot(below_30_yar,below_30_surprised)
        ax4.plot(below_30_yar,below_30_neutral)

        '''ax5.clear()
        ax5.plot(above_30_yar,above_30_sad)
        ax5.plot(above_30_yar,above_30_angry)
        ax5.plot(above_30_yar,above_30_disgust)
        ax5.plot(above_30_yar,above_30_scared)
        ax5.plot(above_30_yar,above_30_happy)
        ax5.plot(above_30_yar,above_30_surprised)
        ax5.plot(above_30_yar,above_30_neutral)'''

        ax1.legend()

    except Exception:
        pass

if os.path.exists('emotion_recorded.txt'):
    os.remove('emotion_recorded.txt')

while not os.path.exists('emotion_recorded.txt'):
    pass

ani = animation.FuncAnimation(fig, animate)
plt.show()
