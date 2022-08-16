import dataLib as DL
import videoLib as VL
import nozzleLibExpNum as NL



names = DL.getFiles("whipping", ".AVI")


VL.convertToMp4(names, 90)


names = DL.getFiles("Nozzle 25_11_21 K/", "mp4")

names1 = names[0:4]
names2 = names[4:8]
names3 = names[8:13]
names4 = names[13:]


#VL.getFrames("Nozzle 25_11_21 K/R1536x1024_G40L1000_F1000.mp4")


video = names1[1]
VL.getFrames(video)
video = names1[2]
VL.getFrames(video)
video = names1[3]
VL.getFrames(video)

ozadje = "output/R1024x768_ozadje_0000.jpg"
merilo = "output/R1024x768_merilo_0000.jpg"
soba = "output/R1024x768_soba_0000.jpg"


a = NL.defineStartJetExp(soba, ozadje, save = True)
c = NL.calcRefExp(merilo, ozadje, 0.68, a, save=True)


NL.measureJetVideoExp(names1[0], ozadje, jet_start=a, mm_to_px_ratio=c[0], locations = [0.01, 0.5, 1.0], start_frame = 0, end_frame = -1, step=1, left_cut = 20, right_cut = 20, next_nth = 5, prev_nth = 5, note='')


video = names2[1]
VL.getFrames(video)
video = names2[2]
VL.getFrames(video)
video = names2[3]
VL.getFrames(video)

ozadje = "output/R1536x1024_ozadje_0000.jpg"
merilo = "output/R1536x1024_merilo_0000.jpg"
soba = "output/R1536x1024_soba_0000.jpg"


a = NL.defineStartJetExp(soba, ozadje, save = True)
c = NL.calcRefExp(merilo, ozadje, 0.68, a, save=True)


NL.measureJetVideoExp(names2[0], ozadje, jet_start=a, mm_to_px_ratio=c[0], locations = [0.01, 0.5, 1.0], start_frame = 0, end_frame = -1, step=1, left_cut = 20, right_cut = 20, next_nth = 5, prev_nth = 5, note='')


video = names3[2]
VL.getFrames(video)
video = names3[3]
VL.getFrames(video)
video = names3[4]
VL.getFrames(video)

ozadje = "output/R512x192_ozadje_0000.jpg"
merilo = "output/R512x192_merilo_0000.jpg"
soba = "output/R512x192_soba_0000.jpg"


a = NL.defineStartJetExp(soba, ozadje, save = True)
c = NL.calcRefExp(merilo, ozadje, 0.68, a, save=True)


NL.measureJetVideoExp(names3[0], ozadje, jet_start=a, mm_to_px_ratio=c[0], locations = [0.016, 0.5, 1.0], start_frame = 0, end_frame = -1, step=1, left_cut = 20, right_cut = 20, next_nth = 5, prev_nth = 5, note='')
NL.measureJetVideoExp(names3[1], ozadje, jet_start=a, mm_to_px_ratio=c[0], locations = [0.016, 0.5, 1.0], start_frame = 0, end_frame = -1, step=1, left_cut = 20, right_cut = 20, next_nth = 5, prev_nth = 5, note='')




video = names4[4]
VL.getFrames(video)
video = names4[5]
VL.getFrames(video)
video = names4[6]
VL.getFrames(video)

ozadje = "output/R768x512_ozadje_0000.jpg"
merilo = "output/R768x512_merilo_0000.jpg"
soba = "output/R768x512_soba_0000.jpg"


a = NL.defineStartJetExp(soba, ozadje, save = True)
c = NL.calcRefExp(merilo, ozadje, 0.68, a, save=True)


NL.measureJetVideoExp(names4[0], ozadje, jet_start=a, mm_to_px_ratio=c[0], locations = [0.01, 0.5, 1.0], start_frame = 0, end_frame = -1, step=1, left_cut = 20, right_cut = 20, next_nth = 5, prev_nth = 5, note='')

NL.measureJetVideoExp(names4[1], ozadje, jet_start=a, mm_to_px_ratio=c[0], locations = [0.01, 0.5, 1.0], start_frame = 0, end_frame = -1, step=1, left_cut = 20, right_cut = 20, next_nth = 5, prev_nth = 5, note='')
NL.measureJetVideoExp(names4[2], ozadje, jet_start=a, mm_to_px_ratio=c[0], locations = [0.01, 0.5, 1.0], start_frame = 0, end_frame = -1, step=1, left_cut = 20, right_cut = 20, next_nth = 5, prev_nth = 5, note='')
NL.measureJetVideoExp(names4[3], ozadje, jet_start=a, mm_to_px_ratio=c[0], locations = [0.01, 0.5, 1.0], start_frame = 0, end_frame = -1, step=1, left_cut = 20, right_cut = 20, next_nth = 5, prev_nth = 5, note='')


   