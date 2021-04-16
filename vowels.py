import numpy as np
import os

'''
Filenames:
character 1:     m=man, w=woman, b=boy, g=girl
characters 2-3:  talker number
characters 4-5:  vowel (ae="had", ah="hod", aw="hawed", eh="head", er="heard",
                        ei="haid", ih="hid", iy="heed", oa=/o/ as in "boat",
                        oo="hood", uh="hud", uw="who'd")

col1:  filename
col2:  duration in msec
col3:  f0 at "steady state"
col4:  F1 at "steady state"
col5:  F2 at "steady state"
col6:  F3 at "steady state"
col7:  F4 at "steady state"
col8:  F1 at 20% of vowel duration
col9:  F2 at 20% of vowel duration
col10: F3 at 20% of vowel duration
col11: F1 at 50% of vowel duration
col12: F2 at 50% of vowel duration
col13: F3 at 50% of vowel duration
col14: F1 at 80% of vowel duration
col15: F2 at 80% of vowel duration
col16: F3 at 80% of vowel duration

'''

vowels = ['ae','ah','aw','eh','er','ei','ih','iy','oa','oo','uh','uw']
talkers = ['m', 'w', 'b', 'g']

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


vowelDataLoc = os.path.join(__location__, 'Data/Vowels/vowdata_nohead.dat')



def main():
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    data = loadData()


def loadData():
    rawData = np.genfromtxt(vowelDataLoc, dtype = str, delimiter=',',)
    for i in rawData:
        print(i)



if __name__ == '__main__':
    main()
