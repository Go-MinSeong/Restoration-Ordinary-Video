#!/bin/bash
#  pip install cupy-cuda111
#  pip install pytube

# make dataset

# python MakeTripletset.py --name "(여자)아이들(G)-IDLE 'TOMBOY (톰보이)' DANCECOVER | 안무 거울모드 | MIRROR_30.0" --start "60" --end "5160" --interval 1
# python MakeTripletset.py --name "[ (여자)아이들((G)I-DLE) - 'Nxde(누드)' 안무 거울모드 MIRRORED | 커버댄스 DANCECOVER | 1인안무 ]_30.0" --start "40" --end "5300" --interval 1
# python MakeTripletset.py --name "[MIRRORED] Brave Girls(브레이브 걸스) - Rollin’ 안무 거울모드_24.0" --start "50" --end "4000" --interval 1
# python MakeTripletset.py --name "[MIRRORED] Hwa Sa(화사) - Maria(마리아) Dance Cover 커버댄스 거울모드 안무_30.0" --start "250" --end "5900" --interval 1
# python MakeTripletset.py --name "뉴진스 NewJeans 'Attention(어텐션)' DANCECOVER | 안무 거울모드 | MIRRORED_30.0" --start "40" --end "3050" --interval 1
# python MakeTripletset.py --name "뉴진스 NewJeans 'Hype boy' DANCECOVER | 안무 거울모드 | MIRRORED_30.0" --start "120" --end "2600" --interval 1
# python MakeTripletset.py --name "뉴진스 NewJeans ‘Ditto' DANCECOVER | MIRRORED | 안무 거울모드_30.0" --start "500" --end "5400" --interval 1
# python MakeTripletset.py --name "다시 만난 세계 안무 커버 [거울모드]_30.0" --start "250" --end "5000" --interval 1
# python MakeTripletset.py --name "LE SSERAFIM (르세라핌) 'ANTIFRAGILE' Dance Cover Mirrored (거울모드)_30.0" --start "90" --end "3600" --interval 1
# python MakeTripletset.py --name "H.O.T - 캔디(Candy) 안무 거울모드_29.97002997002997" --start "90" --end "3300" --interval 1

# python MakeTripletset.py --name "(여자)아이들(G)-IDLE 'TOMBOY (톰보이)' DANCECOVER | 안무 거울모드 | MIRROR_30.0" --start "60" --end "5160" --interval 2
# python MakeTripletset.py --name "[ (여자)아이들((G)I-DLE) - 'Nxde(누드)' 안무 거울모드 MIRRORED | 커버댄스 DANCECOVER | 1인안무 ]_30.0" --start "40" --end "5300" --interval 2
# python MakeTripletset.py --name "[MIRRORED] Brave Girls(브레이브 걸스) - Rollin’ 안무 거울모드_24.0" --start "50" --end "4000" --interval 2
# python MakeTripletset.py --name "[MIRRORED] Hwa Sa(화사) - Maria(마리아) Dance Cover 커버댄스 거울모드 안무_30.0" --start "250" --end "5900" --interval 2
# python MakeTripletset.py --name "뉴진스 NewJeans 'Attention(어텐션)' DANCECOVER | 안무 거울모드 | MIRRORED_30.0" --start "40" --end "3050" --interval 2
# python MakeTripletset.py --name "뉴진스 NewJeans 'Hype boy' DANCECOVER | 안무 거울모드 | MIRRORED_30.0" --start "120" --end "2600" --interval 2
# python MakeTripletset.py --name "뉴진스 NewJeans ‘Ditto' DANCECOVER | MIRRORED | 안무 거울모드_30.0" --start "500" --end "5400" --interval 2
# python MakeTripletset.py --name "다시 만난 세계 안무 커버 [거울모드]_30.0" --start "250" --end "5000" --interval 2
# python MakeTripletset.py --name "LE SSERAFIM (르세라핌) 'ANTIFRAGILE' Dance Cover Mirrored (거울모드)_30.0" --start "90" --end "3600" --interval 2
# python MakeTripletset.py --name "H.O.T - 캔디(Candy) 안무 거울모드_29.97002997002997" --start "90" --end "3300" --interval 2

# python MakeTripletset.py --name "(여자)아이들(G)-IDLE 'TOMBOY (톰보이)' DANCECOVER | 안무 거울모드 | MIRROR_30.0" --start "60" --end "5160" --interval 3
# python MakeTripletset.py --name "[ (여자)아이들((G)I-DLE) - 'Nxde(누드)' 안무 거울모드 MIRRORED | 커버댄스 DANCECOVER | 1인안무 ]_30.0" --start "40" --end "5300" --interval 3
# python MakeTripletset.py --name "[MIRRORED] Brave Girls(브레이브 걸스) - Rollin’ 안무 거울모드_24.0" --start "50" --end "4000" --interval 3
# python MakeTripletset.py --name "[MIRRORED] Hwa Sa(화사) - Maria(마리아) Dance Cover 커버댄스 거울모드 안무_30.0" --start "250" --end "5900" --interval 3
# python MakeTripletset.py --name "뉴진스 NewJeans 'Attention(어텐션)' DANCECOVER | 안무 거울모드 | MIRRORED_30.0" --start "40" --end "3050" --interval 3
# python MakeTripletset.py --name "뉴진스 NewJeans 'Hype boy' DANCECOVER | 안무 거울모드 | MIRRORED_30.0" --start "120" --end "2600" --interval 3
# python MakeTripletset.py --name "뉴진스 NewJeans ‘Ditto' DANCECOVER | MIRRORED | 안무 거울모드_30.0" --start "500" --end "5400" --interval 3
# python MakeTripletset.py --name "다시 만난 세계 안무 커버 [거울모드]_30.0" --start "250" --end "5000" --interval 3
# python MakeTripletset.py --name "LE SSERAFIM (르세라핌) 'ANTIFRAGILE' Dance Cover Mirrored (거울모드)_30.0" --start "90" --end "3600" --interval 3
# python MakeTripletset.py --name "H.O.T - 캔디(Candy) 안무 거울모드_29.97002997002997" --start "90" --end "3300" --interval 3



# image interpolation

# python /home/work/capstone/Final/interpolate_image.py "1"
# python /home/work/capstone/Final/interpolate_image.py "3"
# python /home/work/capstone/Final/interpolate_image.py "5"
# python /home/work/capstone/Final/interpolate_image.py "6"
# python /home/work/capstone/Final/interpolate_image.py "15"
# python /home/work/capstone/Final/interpolate_image.py "18"
# python /home/work/capstone/Final/interpolate_image.py "26"
# python /home/work/capstone/Final/interpolate_image.py "102"
# python /home/work/capstone/Final/interpolate_image.py "108"
# python /home/work/capstone/Final/interpolate_image.py "191"
# python /home/work/capstone/Final/interpolate_image.py "201"
# python /home/work/capstone/Final/interpolate_image.py "204"
# python /home/work/capstone/Final/interpolate_image.py "562"
# python /home/work/capstone/Final/interpolate_image.py "563"
# python /home/work/capstone/Final/interpolate_image.py "2669"
# python /home/work/capstone/Final/interpolate_image.py "2721"
# python /home/work/capstone/Final/interpolate_image.py "2722"
# python /home/work/capstone/Final/interpolate_image.py "2725"
# python /home/work/capstone/Final/interpolate_image.py "3051"
# python /home/work/capstone/Final/interpolate_image.py "3964"
# python /home/work/capstone/Final/interpolate_image.py "3968"
# python /home/work/capstone/Final/interpolate_image.py "4607"
# python /home/work/capstone/Final/interpolate_image.py "4614"
# python /home/work/capstone/Final/interpolate_image.py "5178"
# python /home/work/capstone/Final/interpolate_image.py "6921"
# python /home/work/capstone/Final/interpolate_image.py "6923"
# python /home/work/capstone/Final/interpolate_image.py "10957"
# python /home/work/capstone/Final/interpolate_image.py "10959"
# python /home/work/capstone/Final/interpolate_image.py "11698"
# python /home/work/capstone/Final/interpolate_image.py "12756"
# python /home/work/capstone/Final/interpolate_image.py "19453"
# python /home/work/capstone/Final/interpolate_image.py "19456"
# python /home/work/capstone/Final/interpolate_image.py "45960"
# python /home/work/capstone/Final/interpolate_image.py "45967"
# python /home/work/capstone/Final/interpolate_image.py "49880"
# python /home/work/capstone/Final/interpolate_image.py "baby_walking"
# python /home/work/capstone/Final/interpolate_image.py "baseball"
# python /home/work/capstone/Final/interpolate_image.py "Beanbags"
# python /home/work/capstone/Final/interpolate_image.py "boxing"
# python /home/work/capstone/Final/interpolate_image.py "burnout"
# python /home/work/capstone/Final/interpolate_image.py "choreography"
# python /home/work/capstone/Final/interpolate_image.py "DogDance"
# python /home/work/capstone/Final/interpolate_image.py "e-bike"
# python /home/work/capstone/Final/interpolate_image.py "education"
# python /home/work/capstone/Final/interpolate_image.py "finegym"
# python /home/work/capstone/Final/interpolate_image.py "inflatable"
# python /home/work/capstone/Final/interpolate_image.py "juggle"
# python /home/work/capstone/Final/interpolate_image.py "jump"
# python /home/work/capstone/Final/interpolate_image.py "jump_rope"
# python /home/work/capstone/Final/interpolate_image.py "kart-turn"
# python /home/work/capstone/Final/interpolate_image.py "kids-turning"
# python /home/work/capstone/Final/interpolate_image.py "monkeys"
# python /home/work/capstone/Final/interpolate_image.py "swing-boy"
# python /home/work/capstone/Final/interpolate_image.py "tackle"
# python /home/work/capstone/Final/interpolate_image.py "varanus-tree"
# python /home/work/capstone/Final/interpolate_image.py "what"


# video interpolation

# #Ditto
#python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=LlBG2ipO6ZU" --frame 10
#python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=LlBG2ipO6ZU"
# #캔디
#python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=_Y7wlTk1-XQ&t=2s" --frame 10
#python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=_Y7wlTk1-XQ&t=2s"
# #ANTIFRAGILE
#python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=aTh-FjI5__Q" --frame 10
#python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=aTh-FjI5__Q"
# # 하입보이~ f
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=eIBVFbrvpng"
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=eIBVFbrvpng" --frame 10
# # # 톰보이
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=qJTAGJBVmys"
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=qJTAGJBVmys" --frame 10
# # # OMG
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=NLZVOCpHkak"
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=NLZVOCpHkak" --frame 10
# # # poppy
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=34ALUxfHBtM"
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=34ALUxfHBtM" --frame 10
# # # 마리아
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=Z9Sn9r82gyE"
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=Z9Sn9r82gyE" --frame 10
# # # 다시 만난 세계
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=wWA3ICLkSD4&t=8s"
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=wWA3ICLkSD4&t=8s" --frame 10
# # # Nxde
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=jH8OIsiIHoI"
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=jH8OIsiIHoI" --frame 10
# # # Attention
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=7Eg2vSjAAFk"
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=7Eg2vSjAAFk" --frame 10
# # # Rollin
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=EMQaCq-S-LE"
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=EMQaCq-S-LE" --frame 10

