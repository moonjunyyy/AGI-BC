import os
import pandas as pd
import torch
import torchvision
import torchaudio
from torch.utils.data import Dataset
from typing import Callable
import math
import datetime

df = pd.read_csv("/data/datasets/ETRI_resize/etri.tsv", sep='\t', encoding='utf-8')
print(df)

# file_name = [
#     "",
#     "220918_남정희_김우진",
#     "220918_남정희_김지수",
#     "220918_남정희_백보경",
#     "220918_남정희_손석규",
#     "220918_남정희_정정연",
#     "220918_남정희_차지수",
#     "220920_강명진_오주현",
#     "220920_강명진_윤수진",
#     "220920_강명진_정은영",
#     "220920_강명진_조주현",
#     "220922_강명진_오은숙",
#     "220922_강명진_정준호",
#     "220922_강주영_강태랑",
#     "220922_강주영_김은영",
#     "220922_강주영_이준혁",
#     "220922_강주영_정유림",
#     "220922_강주영_최보규",
#     "220925_남정희_김민석",
#     "220925_남정희_박종길",
#     "220925_남정희_서지원",
#     "220925_남정희_서혜연",
#     "220925_남정희_이주왕",
#     "220925_남정희_장은태",
#     "220925_남정희_한성민",
#     "220925_남정희_허세민",
#     "220929_강명진_김민수",
#     "220929_강명진_김정현",
#     "220929_강명진_류호정",
#     "220929_강명진_유채이",
#     "220929_강주영_김영미",
#     "220929_강주영_류서영",
#     "220929_강주영_송선희",
#     "220929_강주영_임지윤",
#     "221006_윤지선_박일용",
#     "221006_윤지선_안수진",
#     "221006_윤지선_용금여",
#     "221006_윤지선_임현숙",
#     "221006_윤지선_조영현",
#     "221006_윤지선_채원석",
#     "221006_윤지선_최주희",
# ]


# for i in range(len(file_name)):
#     os.system(f'mkdir /data/datasets/ETRI_resize/{i}')

# for i in range(len(file_name)):
#     path = os.path.join("/data/datasets/ETRI_Backchannel_Corpus_2022", file_name[i], "'1. 상담자 녹음본.wav'")
#     os.system(f'cp {path} /data/datasets/ETRI_resize/{i}/counselor.wav')
#     path = os.path.join("/data/datasets/ETRI_Backchannel_Corpus_2022", file_name[i], "'2. 내담자 녹음본.wav'")
#     os.system(f'cp {path} /data/datasets/ETRI_resize/{i}/client.wav')
#     path = os.path.join("/data/datasets/ETRI_Backchannel_Corpus_2022", file_name[i], '"4. 상담자 인터뷰 영상.mp4"' if i > 10 else '"5. 상담자 인터뷰 영상.mp4"')
#     os.system(f'ffmpeg -i {path} -filter:v "crop=640:640:640:0,scale=224:224" /data/datasets/ETRI_resize/{i}/counselor.mp4')
#     path = os.path.join("/data/datasets/ETRI_Backchannel_Corpus_2022", file_name[i], '"5. 내담자 인터뷰 영상.mp4"' if i > 10 else '"6. 내담자 인터뷰 영상.mp4"')
#     os.system(f'ffmpeg -i {path} -filter:v "crop=640:640:640:0,scale=224:224" /data/datasets/ETRI_resize/{i}/client.mp4')

# badfile = [447, 449, 450, 451, 454, 455, 456, 458, 461, 1758, 1760, 1762, 1764, 1765, 1767, 1768, 1769, 1771, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1781, 1782, 1786, 1788, 1790, 1791, 1792, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1872, 2156, 2534, 2535, 2536, 2537, 2538, 2539, 3432, 3433, 3435, 3437, 3440, 3442, 3450, 3451, 3453, 3455, 3457, 3459, 3461, 3463, 3465, 3472, 3473, 3476, 3478, 3480, 3482, 3485, 3491, 3493, 3496, 3497, 3498, 5815, 5816, 5817, 5818, 5819, 6146, 6147, 6148, 6151, 6152, 6154, 6155, 6156, 6157, 6158, 6159, 6160, 6162, 6163, 6166, 6167, 6168, 6170, 6171, 6172, 6175, 6177, 7073, 7075, 7080, 7430, 7435, 7436, 7437, 7438, 7439, 7440, 7443, 7740, 7741, 7743, 7745, 7746, 7747, 7749, 7750, 7752, 7753, 7754, 7755, 7756, 7757, 7758, 7759, 7760, 7762, 7764, 8978, 8980, 8982, 8983, 8984, 8985, 10443, 10444, 10445, 10447, 10448, 10449, 10451, 10453, 10454, 10457, 10460, 10461, 10462, 10463, 10464, 10465, 10466, 10468, 10471, 10474, 10779, 10782, 10785, 10786, 10792, 10793, 10795, 10796, 10797, 10799, 10800, 10801, 10804, 10805, 10811, 10812, 10813, 10815, 10816, 10817, 10818, 10819, 10821, 10822, 10824, 10825, 10826, 10827, 10828, 10831, 10835, 10838, 10839, 10840, 10842, 10843, 10845, 10848, 10849, 10850, 10851, 10852, 10853, 10855, 10856, 10857, 10858, 10859, 10860, 10861, 10862, 10863, 10864, 10866, 10870, 10871, 10874, 10875, 10877, 10878, 10879, 10880, 10881, 10882, 10884, 10885, 10886, 10887, 10890, 10891, 10892, 10894, 10895, 10896, 11798, 11799, 11800, 11802, 11807, 11809, 11811, 11814, 11816, 11818, 11821, 11822, 11824, 11829, 11830, 11831, 11835, 11837, 11841, 11842, 11843, 11844, 11845, 11846, 11850, 11852, 11854, 11857, 11859, 11861, 11862, 11864, 11867, 11868, 12078, 12080, 12081, 12082, 12085, 12086, 12088, 12089, 12090, 12091, 12092, 12093, 12094, 12097, 12098, 12101, 12102, 12103, 12106, 12107, 12108, 12109, 12110, 12111, 12112, 12113, 12114, 12118, 12122, 12124, 12127, 12128, 12131, 12132, 12133, 12134, 12135, 12137, 12138, 12139, 12141, 12142, 12146, 12147, 12148, 12150, 12151, 12153, 12154, 12155, 12156, 12158, 12161, 12164, 12166, 12167, 12168, 12170, 12173, 12174, 12177, 12179, 12180, 12182, 12184, 12185, 12186, 12187, 12188, 12191, 12193, 12194, 12195, 12196, 12197, 12199, 12201, 12203, 12205, 12209, 12211, 12213, 12215, 12218, 12220, 12221, 12227, 12229, 12234, 12236, 12241, 12243, 12246, 12248, 12249, 12250, 12255, 12256, 12258, 12260, 12263, 12264, 12265, 12266, 12267, 12269, 12276, 12280, 12281, 12285, 12287, 12289, 12291, 12293, 12294, 12297, 12299, 12303, 12305, 12306, 12307, 12308, 12309, 12312, 12315, 12318, 12319, 12323, 12330, 12331, 12335, 12337, 12344, 12346, 12347, 12349, 12351, 12353, 12354, 12361, 12364, 12368, 12374, 12387, 12394, 12396, 12397, 12399, 12402, 12403, 12404, 12408, 12410, 12415, 12416, 12419, 12420, 12422, 12424, 12426, 12428, 12430, 12433, 12434, 12435, 12441, 12442, 12445, 12447, 12449, 12450, 12455, 12457, 12459, 12461, 12464, 12466, 12469, 12470, 12472, 12474, 12478, 12480, 12483, 12488, 12490, 12492, 12493, 12496, 12497, 12498, 12499, 12501, 12502, 12503, 12504, 12506, 12509, 12510, 12511, 12519, 12521, 12523, 12524, 12525, 12534, 12535, 12538, 12539, 12540, 12543, 12544, 12547, 12548, 12550, 12551, 12552, 12553, 12554, 12555, 12557, 12558, 12559, 12560, 12561, 12563, 12564, 12566, 12568, 12569, 12570, 12573, 12575, 12576, 12577, 12578, 12579, 12580, 12581, 12584, 12586, 12587, 12589, 12592, 12593, 12595, 12596, 12597, 12598, 12599, 12600, 12602, 12603, 12604, 12605, 12607, 12608, 12609, 12610, 12611, 12612, 12613, 12614, 12616, 12617, 12618, 12619, 12621, 12622, 12624, 12625, 12626, 12627, 12629, 12630, 12631, 12632, 12633, 12634, 12635, 12638, 12639, 12640, 12645, 12648, 12649, 12650, 12651, 12652, 12653, 12654, 12656, 12657, 12658, 12659, 12660, 12661, 12662, 12663, 12664, 12665, 12666, 12667, 12668, 12669, 12671, 12672, 12673, 12674, 12675, 12676, 12678, 12679, 12680, 12681, 12682, 12683, 12684, 12685, 12686, 12688, 12689, 12691, 12692, 12693, 12694, 12699, 12700, 12703, 12704, 12706, 12707, 12708, 12709, 12710, 12711, 12712, 12713, 12716, 12718, 12719, 12721, 12722, 12723, 12724, 12726, 12727, 12730, 12733, 12734, 12735, 12737, 12741, 12742, 12744, 12745, 12746, 12748, 12749, 12750, 12752, 12754, 12756, 12757, 12759, 12760, 12761, 12763, 12765, 12766, 12768, 12769, 12770, 12771, 12772, 12774, 12775, 12777, 12779, 12781, 12783, 12785, 12788, 12790, 12791, 12793, 12796, 12798, 12800, 12802, 12804, 12806, 12808, 12810, 12811, 12813, 12815, 12817, 12818, 12820, 12823, 12825, 12827, 12829, 12831, 12834, 12836, 12838, 12840, 12843, 12844, 12846, 12849, 12852, 12855, 12859, 12861, 12863, 12864, 12866, 12868, 12870, 12872, 12873, 12875, 12879, 12880, 12882, 12884, 12886, 12887, 12889, 12891, 12893, 12897, 12898, 12899, 12901, 12903, 12905, 12906, 12909, 12911, 12912, 12913, 12915, 12917, 12918, 12924, 12927, 12929, 12931, 12933, 12935, 12937, 12939, 12941, 12943, 12945, 12947, 12949, 12950, 12955, 12957, 12959, 12961, 12963, 12965, 12967, 12969, 12971, 12972, 12974, 12975, 12977, 12979, 12980, 12982, 12984, 12985, 12987, 12988, 12990, 12992, 12994, 12996, 12998, 13000, 13002, 13003, 13005, 13007, 13009, 13013, 13016, 13018, 13021, 13023, 13025, 13027, 13029, 13031, 13032, 13034, 13035, 13042, 13045, 13049, 13050, 13051, 13052, 13055, 13062, 13063, 13065, 13067, 13068, 13073, 13074, 13075, 13076, 13078, 13080, 13081, 13085, 13092, 13095, 13098, 13100, 13109, 13114, 13115, 13118, 13119, 13123, 13132, 13135, 13141, 13142, 13143, 13147, 13152, 13165, 13166, 13171, 13173, 13176, 13179, 13184, 13185, 13186, 13187, 13190, 13199, 13202, 13209, 13216, 13226, 13228, 13231, 13233, 13238, 13243, 13248, 13260, 13268, 13269, 13272, 13275, 13284, 13285, 13286, 13288, 13290, 13291, 13295, 13297, 13299, 13301, 13303, 13304, 13307, 13308, 13310, 13311, 13312, 13313, 13314, 13315, 13316, 13317, 13318, 13319, 13321, 13322, 13323, 13324, 13325, 13326, 13327, 13330, 13331, 13332, 13333, 13334, 13338, 13339, 13340, 13341, 13342, 13343, 13344, 13345, 13346, 13347, 13348, 13349, 13350, 13351, 13352, 13353, 13354, 13355, 13356, 13357, 13358, 13359, 13360, 13362, 13365, 13366, 13367, 13368, 13369, 13370, 13371, 13372, 13373, 13374, 13375, 13377, 13379, 13380, 13382, 13383, 13384, 13385, 13386, 13387, 13388, 13390, 13391, 13392, 13393, 13395, 13397, 13398, 13399, 13400, 13401, 13403, 13404, 13406, 13407, 13408, 13409, 13410, 13411, 13412, 13414, 13416, 13421, 13422, 13424, 13427, 13429, 13431, 13432, 13436, 13437, 13439, 13440, 13442, 13443, 13444, 13445, 13446, 13447, 13448, 13449, 13451, 13452, 13455, 13456, 13457, 13458, 13460, 13465, 13466, 13467, 13469, 13470, 13471, 13473, 13474, 13476, 13478, 13479, 13480, 13481, 13482, 13483, 13484, 13485, 13487, 13489, 13490, 13492, 13493, 13495, 13496, 13497, 13499, 13500, 13503, 13504, 13505, 13507, 13509, 13510, 13512, 13514, 13515, 13516, 13518, 13519, 13520, 13521, 13522, 13524, 13525, 13526, 13527, 13528, 13530, 13533, 13534, 13535, 13536, 13537, 13538, 13539, 13542, 13543, 13544, 13545, 13546, 13548, 13550, 13552, 13553, 13554, 13555, 13556, 13557, 13558, 13561, 13567, 13572, 13576, 13577, 13583, 13584, 13589, 13590, 13591, 13594, 13595, 13596, 13598, 13602, 13604, 13606, 13611, 13613, 13618, 13620, 13621, 13627, 13630, 13634, 13635, 13638, 13644, 13646, 13647, 13649, 13653, 13654, 13660, 13661, 13663, 13664, 13666, 13667, 13668, 13669, 13672, 13674, 13677, 13678, 13679, 13685, 13686, 13693, 13694, 13696, 13698, 13699, 13702, 13703, 13704, 13706, 13707, 13709, 13710, 13711, 13712, 13713, 13714, 13715, 13716, 13717, 13719, 13720, 13726, 13732, 13734, 13740, 13743, 13744, 13746, 13748, 13751, 13754, 13757, 13758, 13759, 13761, 13762, 13766, 13767, 13769, 13772, 13773, 13774, 13777, 13778, 13779, 13780, 13782, 13784, 13785, 13788, 13789, 13791, 13792, 13793, 13794, 13800, 13802, 13803, 13805, 13806, 13809, 13810, 13811, 13812, 13814, 13815, 13816, 13817, 13818, 13820, 13821, 13822, 13824, 13826, 13832, 13834, 13839, 13841, 13844, 13846, 13849, 13850, 13853, 13854, 13855, 13857, 13858, 13860, 13861, 13863, 13864, 13866, 13874, 13877, 13878, 13880, 13882, 13884, 13885, 13887, 13888, 13889, 13890, 13891, 13892, 13894, 13895, 13896, 13897, 13898, 13899, 13902, 13903, 13905, 13906, 13907, 13909, 13910, 13914, 13917, 13920, 13921, 13923, 13925, 13926, 13928, 13929, 13930, 13931, 13932, 13935, 13937, 13939, 13941, 13943, 13945, 13946, 13948, 13950, 13951, 13953, 13955, 13956, 13959, 13961, 13966, 13968, 13969, 13974, 13976, 13978, 13979, 13981, 13983, 13985, 13988, 13989, 13992, 13994, 13995, 13996, 13997, 13998, 14000, 14002, 14004, 14005, 14006, 14007, 14010, 14015, 14016, 14017, 14018, 14022, 14024, 14025, 14026, 14027, 14029, 14032, 14033, 14035, 14036, 14037, 14038, 14039, 14041, 14042, 14043, 14046, 14050, 14053, 14055, 14056, 14059, 14061, 14063, 14064, 14066, 14067, 14069, 14070, 14072, 14073, 14075, 14077, 14079, 14081, 14083, 14084, 14085, 14087, 14088, 14090, 14091, 14092, 14093, 14094, 14096, 14098, 14104, 14105, 14106, 14108, 14110, 14112, 14113, 14116, 14118, 14120, 14122, 14124, 14126, 14127, 14129, 14130, 14132, 14134, 14135, 14136, 14137, 14138, 14142, 14143, 14144, 14145, 14146, 14147, 14149, 14151, 14154, 14155, 14157, 14158, 14161, 14162, 14164, 14166, 14167, 14168, 14170, 14171, 14172, 14174, 14175, 14177, 14179, 14181, 14183, 14184, 14186, 14188, 14190, 14192, 14193, 14195, 14198, 14201, 14203, 14204, 14209, 14210, 14211, 14213, 14216, 14218, 14219, 14221, 14222, 14224, 14225, 14227, 14228, 14231, 14233, 14235, 14236, 14237, 14238, 14240, 14241, 14243, 14244, 14246, 14248, 14249, 14251, 14252, 14254, 14255, 14257, 14258, 14260, 14261, 14263, 14265, 14266, 14267, 14269, 14271, 14272, 14274, 14275, 14277, 14278, 14280, 14282, 14283, 14285, 14286, 14287, 14290, 14291, 14292, 14294, 14296, 14297, 14299, 14300, 14304, 14308, 14310, 14312, 14315, 14317, 14319, 14320, 14322, 14326, 14327, 14328, 14329, 14330, 14332, 14334, 14336, 14337, 14339, 14341, 14343, 14345, 14346, 14348, 14349, 14352, 14354, 14355, 14357, 14360, 14362, 14364, 14365, 14367, 14368, 14370, 14372, 14375, 14376, 14377, 14379, 14381, 14383, 14385, 14387, 14388, 14389, 14392, 14395, 14397, 14399, 14401, 14403, 14405, 14406, 14407, 14409, 14413, 14415, 14416, 14418, 14419, 14422, 14427, 14429, 14431, 14433, 14434, 14437, 14439, 14441, 14442, 14443, 14445, 14447, 14449, 14451, 14452, 14455, 14457, 14458, 14459, 14460, 14463, 14468, 14470, 14472, 14474, 14475, 14477, 14478, 14479, 14480, 14482, 14484, 14486, 14487, 14488, 14490, 14492, 14493, 14495, 14497, 14500, 14501, 14502, 14503, 14504, 14505, 14506, 14507, 14508, 14509, 14511, 14512, 14515, 14516, 14517, 14518, 14520, 14521, 14523, 14524, 14525, 14528, 14530, 14531, 14532, 14533, 14534, 14535, 14536, 14537, 14540, 14541, 14542, 14543, 14546, 14548, 14549, 14551, 14552, 14555, 14556, 14558, 14559, 14560, 14562, 14563, 14566, 14567, 14570, 14574, 14575, 14576, 14578, 14579, 14582, 14583, 14585, 14586, 14588, 14589, 14591, 14592, 14593, 14594, 14595, 14596, 14597, 14598, 14599, 14600, 14601, 14602, 14603, 14604, 14605, 14607, 14608, 14609, 14612, 14613, 14614, 14616, 14617, 14621, 14623, 14625, 14626, 14628, 14629, 14630, 14631, 14632, 14634, 14635, 14636, 14637, 14640, 14642, 14643, 14644, 14645, 14646, 14648, 14649, 14650, 14651, 14653, 14654, 14657, 14660, 14663, 14664, 14665, 14666, 14667, 14668, 14669, 14670, 14672, 14675, 14676, 14677, 14680, 14681, 14682, 14685, 14686, 14687, 14688, 14690, 14691, 14692, 14694, 14695, 14698, 14699, 14700, 14701, 14703, 14704, 14705, 14706, 14707, 14708, 14709, 14710, 14716, 14717, 14718, 14719, 14720, 14721, 14724, 14725, 14726, 14727, 14728, 14729, 14731, 14732, 14733, 14734, 14735, 14736, 14737, 14738, 14740, 14741, 14742, 14743, 14744, 14745, 14746, 14747, 14748, 14749, 14750, 14751, 14752, 14753, 14754, 14755, 14757, 14759, 14760, 14761, 14762, 14763, 14764, 14765]

badfile = []

for idx, row in df.iterrows():
    # print(idx)
    if len(badfile) != 0:
        if idx not in badfile:
            continue
    start = row['start']
    end   = row['end']

    if start > end:
        start, end = end, start

    start = str(datetime.timedelta(seconds=start))
    end = str(datetime.timedelta(seconds=end))
    
    path = f"/data/datasets/ETRI_resize/{row['folder']}/{row['role']}.mp4"
    print(f'ffmpeg -y -i {path} -ss {start} -to {end} -c copy /data/datasets/ETRI_resize/{idx}.mp4')
    os.system(f'ffmpeg -y -i {path} -ss {start} -to {end} /data/datasets/ETRI_resize/{idx}.mp4')
    # os.system(f'ffmpeg -i {path} -ss {start} -to {end} -c copy /data/datasets/ETRI_resize/{idx}.mp4')

    path = f"/data/datasets/ETRI_resize/{row['folder']}/{row['role']}.wav"
    print(f'ffmpeg -y -i {path} -ss {start} -to {end} -c copy /data/datasets/ETRI_resize/{idx}.wav')
    os.system(f'ffmpeg -y -i {path} -ss {start} -to {end} /data/datasets/ETRI_resize/{idx}.wav')
    # if idx == 1:
    #     break