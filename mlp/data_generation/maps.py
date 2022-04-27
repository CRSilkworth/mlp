import numpy as np


all_data_types = np.array([0, 1, 2, 3])
all_languages = np.array(['en', 'ja', 'ko', 'zh_hans', 'zh_hant'])
all_caps = np.array([True, False])
all_abbr = np.array([True, False])
all_era = np.array([True, False])
all_zeros = np.array([True, False])
all_no_year = np.array([True, False])
all_postfix = np.array([True, False])
all_delimiters = np.array(['-', '/', '_', ' ', ''])
all_next = np.array(['this', 'next', 'last'])
all_digit_types = np.array(['word', 'alt_word_0', 'full_width', 'digit'])
all_num_days = np.array([-2, -1, 0, 1, 2])
am_pm_to_tuple = {
  'en': {
    'am': (0, 0, 0),
    'pm': (12, 0, 0),
    'AM': (0, 0, 0),
    'PM': (12, 0, 0),
    ' am': (0, 0, 0),
    ' pm': (12, 0, 0),
    ' AM': (0, 0, 0),
    ' PM': (12, 0, 0),
    ' in the morning': (0, 0, 0),
    ' at night': (12, 0, 0),
    ' in the afternoon': (12, 0, 0),
    ' in the evening': (12, 0, 0),
    '': (0, 0, 0),

  },
  'ja': {
    '日中の': (12, 0, 0),
    '日中': (12, 0, 0),
    '午後': (12, 0, 0),
    '夕方': (12, 0, 0),
    '朝': (0, 0, 0),
    '夜': (12, 0, 0),
    '午前': (12, 0, 0),
    '深夜': (0, 0, 0),
    'PM': (12, 0, 0),
    'AM': (0, 0, 0),
    'お昼の': (12, 0, 0),
    '昼の': (12, 0, 0),
    'お昼': (12, 0, 0),
    '昼': (12, 0, 0),
    '': (0, 0, 0),


  },
  'zh_hans': {
    '凌晨': (0, 0, 0),
    '深夜': (0, 0, 0),
    '凌晨': (0, 0, 0),
    '清晨的': (0, 0, 0),
    '清晨': (0, 0, 0),
    '早上': (0, 0, 0),
    '早晨': (0, 0, 0),
    '早晨': (0, 0, 0),
    '下午': (12, 0, 0),
    '下午的': (12, 0, 0),
    '晚上': (12, 0, 0),
    '傍晚': (12, 0, 0),
    '晚上': (12, 0, 0),
    '点深夜': (12, 0, 0),
    '点晚上': (12, 0, 0),
    '': (0, 0, 0),

  },
  'zh_hant': {
    '凌晨': (0, 0, 0),
    '深夜': (0, 0, 0),
    '凌晨': (0, 0, 0),
    '清晨的': (0, 0, 0),
    '清晨': (0, 0, 0),
    '早上': (0, 0, 0),
    '早晨': (0, 0, 0),
    '早晨': (0, 0, 0),
    '下午': (12, 0, 0),
    '下午的': (12, 0, 0),
    '晚上': (12, 0, 0),
    '傍晚': (12, 0, 0),
    '晚上': (12, 0, 0),
    '点深夜': (12, 0, 0),
    '点晚上': (12, 0, 0),
    '': (0, 0, 0),

  },
  'ko': {
    '오전': (0, 0, 0),
    '새벽': (0, 0, 0),
    '아침 ': (0, 0, 0),
    '오후': (12, 0, 0),
    '낮': (12, 0, 0),
    '저녁': (12, 0, 0),
    '밤': (12, 0, 0),
    '': (0, 0, 0),

  }
}

all_am_pm_types = np.array(list(set(
  list(am_pm_to_tuple['en'].keys()) +
  list(am_pm_to_tuple['ja'].keys()) +
  list(am_pm_to_tuple['zh_hans'].keys()) +
  list(am_pm_to_tuple['zh_hant'].keys()) +
  list(am_pm_to_tuple['ko'].keys()))))

hour_words = {
  'en': ['oclock', "o'clock", ' oclock', " o'clock", ':', ' ', '', ],
  'ja': ['時', ':', '', ],
  'zh_hans': ['点', ':', '', ],
  'zh_hant': ['點', ':', '', ],
  'ko': ['시', ':', '', ]
}
all_hour_words = (
  list(hour_words['en']) +
  list(hour_words['ja']) +
  list(hour_words['zh_hans']) +
  list(hour_words['zh_hant']) +
  list(hour_words['ko']))

minute_words = {
  'en': [':', '.', ' ', ''],
  'ja': [':', '.', ' ', '分', ' ', ''],
  'zh_hans': [':', '.', '分', ' ', ''],
  'zh_hant': [':', '.', '分', ' ', ''],
  'ko': [':', '.', '분', ' ', ''],
}

all_minute_words = (
  list(minute_words['en']) +
  list(minute_words['ja']) +
  list(minute_words['zh_hans']) +
  list(minute_words['zh_hant']) +
  list(minute_words['ko']))

short_cut_words = {
  'en': {
    'half past ': (0, 30, 0),
    'quarter past ': (0, 15, 0),
    '': (0, 0, 0),

  },
  'ja': {
    '半': (0, 30, 0),
    '': (0, 0, 0),

  },
  'zh_hans': {
    '半': (0, 30, 0),
    '': (0, 0, 0),

  },
  'zh_hant': {
    '半': (0, 30, 0),
    '': (0, 0, 0),

  },
  'ko': {
    '半': (0, 30, 0),
    '': (0, 0, 0),
  },


}

all_short_cut_words = (
  list(short_cut_words['en'].keys()) +
  list(short_cut_words['ja'].keys()) +
  list(short_cut_words['zh_hans'].keys()) +
  list(short_cut_words['zh_hant'].keys()) +
  list(short_cut_words['ko'].keys()))


word_to_tuple = {
  'en': {
    'noon': (12, 0, 0),
    'midnight': (0, 0, 0),
    'midday': (12, 0, 0),
    'mid-day': (12, 0, 0),
    'mid day': (12, 0, 0),
  },
  'ja': {
    '深夜': (0, 0, 0),
    '昼': (12, 0, 0),

  },

  'zh_hans': {
    '午夜': (0, 0, 0),
    '凌晨': (0, 0, 0),
    '深夜': (0, 0, 0),
    '中午': (12, 0, 0),
    '正中午': (12, 0, 0)
  },

  'zh_hant': {
    '午夜': (0, 0, 0),
    '凌晨': (0, 0, 0),
    '深夜': (0, 0, 0),
    '中午': (12, 0, 0),
    '正中午': (12, 0, 0)
  },

  'ko': {
   '밤 12시': (0, 0, 0),
   '자정': (0, 0, 0),
   '자정 12시': (0, 0, 0),
   '밤 열두 시': (0, 0, 0),
   '오전 12시': (0, 0, 0),
   '0시': (0, 0, 0),
  }

}

digit_to_word = {
  'en': {
    0: {'word': 'zero'},
    1: {'word': 'one', 'alt_word_0': 'a'},
    2: {'word': 'two'},
    3: {'word': 'three'},
    4: {'word': 'four'},
    5: {'word': 'five'},
    6: {'word': 'six'},
    7: {'word': 'seven'},
    8: {'word': 'eight'},
    9: {'word': 'nine'},
    10: {'word': 'ten'},
    11: {'word': 'eleven'},
    12: {'word': 'twelve'},
    13: {'word': 'thirteen'},
    14: {'word': 'fourteen'},
    15: {'word': 'fifteen'},
    16: {'word': 'sixteen'},
    17: {'word': 'seventeen'},
    18: {'word': 'eighteen'},
    19: {'word': 'nineteen'},
    20: {'word': 'twenty'},
    30: {'word': 'thirty'},
    40: {'word': 'forty'},
    50: {'word': 'fifty'},
    60: {'word': 'sixty'},
    70: {'word': 'seventy'},
    80: {'word': 'eighty'},
    90: {'word': 'ninety'},
    100: {'word': 'hundred'},
    1000: {'word': 'thousand'},
    1000000: {'word': 'million'},
    int(1e9): {'word': 'billion'},
    int(1e12): {'word': 'trillion'},
    int(1e15): {'word': 'quadrillion'},
    int(1e18): {'word': 'quintillion'},
    int(1e21): {'word': 'sextillion'},
    int(1e24): {'word': 'septillion'},
    int(1e27): {'word': 'octillion'},
    int(1e30): {'word': 'nonillion'},
    int(1e33): {'word': 'decillion'}
  },
  'ja': {
    0: {'word': '〇'},
    1: {'word': '一', 'alt_word_0': '壱'},
    2: {'word': '二', 'alt_word_0': '弐'},
    3: {'word': '三', 'alt_word_0': '参'},
    4: {'word': '四'},
    5: {'word': '五'},
    6: {'word': '六'},
    7: {'word': '七'},
    8: {'word': '八'},
    9: {'word': '九'},
    10: {'word': '十'},
    11: {'word': '十一'},
    12: {'word': '十二'},
    100: {'word': '百'},
    1000: {'word': '千'},
    10000: {'word': '万'},
    100000000: {'word': '億'},
    1000000000000: {'word': '兆'},
    10000000000000000: {'word': '京'},
  },
  'ko': {
    0: {'word': '영'},
    1: {'word': '일'},
    2: {'word': '이'},
    3: {'word': '삼'},
    4: {'word': '사'},
    5: {'word': '오'},
    6: {'word': '육'},
    7: {'word': '칠'},
    8: {'word': '팔'},
    9: {'word': '구'},
    10: {'word': '십'},
    11: {'word': '십일'},
    12: {'word': '십이'},
    100: {'word': '백'},
    1000: {'word': '천'},
    10000: {'word': '만'},
    100000000: {'word': '억'},
    int(1e12): {'word': '조'},
    int(1e16): {'word': '경'},
    int(1e20): {'word': '해'}
  },
  'zh_hans': {
    0: {'word': '〇'},
    1: {'word': '一', 'alt_word_0': '壱'},
    2: {'word': '二', 'alt_word_0': '弐'},
    3: {'word': '三', 'alt_word_0': '参'},
    4: {'word': '四'},
    5: {'word': '五'},
    6: {'word': '六'},
    7: {'word': '七'},
    8: {'word': '八'},
    9: {'word': '九'},
    10: {'word': '十'},
    11: {'word': '十一'},
    12: {'word': '十二'},
    100: {'word': '百'},
    1000: {'word': '千'},
    10000: {'word': '万'},
    100000000: {'word': '億'},
    1000000000000: {'word': '兆'},
    10000000000000000: {'word': '京'},
  },
  'zh_hant': {
    0: {'word': '〇'},
    1: {'word': '一', 'alt_word_0': '壱'},
    2: {'word': '二', 'alt_word_0': '弐'},
    3: {'word': '三', 'alt_word_0': '参'},
    4: {'word': '四'},
    5: {'word': '五'},
    6: {'word': '六'},
    7: {'word': '七'},
    8: {'word': '八'},
    9: {'word': '九'},
    10: {'word': '十'},
    11: {'word': '十一'},
    12: {'word': '十二'},
    100: {'word': '百'},
    1000: {'word': '千'},
    10000: {'word': '万'},
    100000000: {'word': '億'},
    1000000000000: {'word': '兆'},
    10000000000000000: {'word': '京'},
  }

}

_half_to_full = {
  '0': '０',
  '1': '１',
  '2': '２',
  '3': '３',
  '4': '４',
  '5': '５',
  '6': '６',
  '7': '７',
  '8': '８',
  '9': '９',
}


def half_to_full(half):
  half_str = str(half)
  full_str = ''
  for s in half_str:
    full_str = full_str + _half_to_full[s]
  return full_str


word_to_digit = {}
all_digit_types = set()
for lang in digit_to_word:
  word_to_digit[lang] = {}
  for num in digit_to_word[lang]:
    digit_to_word[lang][num]['digit'] = str(num)
    digit_to_word[lang][num]['full_width'] = half_to_full(num)

    sorted_words = [(word_type, word) for word_type, word in digit_to_word[lang][num].items()]
    sorted_words.sort(key=lambda k: -len(k[1]))
    for word_type, word in sorted_words:
      word_to_digit[lang][word] = {'word_type': word_type, 'integer': num}
    for digit_type_key in digit_to_word[lang][num]:
      all_digit_types.add(digit_type_key)

all_digit_types = list(all_digit_types)


month_to_string = {}
for language in all_languages:
  month_to_string[language] = {}
  for month_num in range(1, 13, 1):
    month_to_string[language][month_num] = {}

month_to_string['en'] = {
  1: {'word': 'january', 'abbr': 'jan'},
  2: {'word': 'february', 'abbr': 'feb'},
  3: {'word': 'march', 'abbr': 'mar'},
  4: {'word': 'april', 'abbr': 'apr'},
  5: {'word': 'may', 'abbr': 'may'},
  6: {'word': 'june', 'abbr': 'jun'},
  7: {'word': 'july', 'abbr': 'jul'},
  8: {'word': 'august', 'abbr': 'aug'},
  9: {'word': 'september', 'abbr': 'sep'},
  10: {'word': 'october', 'abbr': 'oct'},
  11: {'word': 'november', 'abbr': 'nov'},
  12: {'word': 'december', 'abbr': 'dec'}
}

month_words = {
  'ja': {'word': '月'},
  'ko': {'word': '월'}
}
month_words['zh_hans'] = month_words['ja']
month_words['zh_hant'] = month_words['ja']


for month_num in range(1, 13, 1):
  for lang in ['ja', 'zh_hans', 'zh_hant']:
    for word_type in ['word', 'digit', 'full_width']:
      month_to_string[lang][month_num]['word'] = digit_to_word[lang][month_num][word_type] + month_words[lang]['word']

string_to_month = {}
for lang in month_to_string:
  string_to_month[lang] = {}
  for month_num, strings in month_to_string[lang].items():
    for word_type, word in strings.items():
      string_to_month[lang][word] = {'word_type': word_type, 'month': month_num}

ja_era_to_offset = {'文亀': 1500, '永正': 1503, '大永': 1520, '享禄': 1527, '天文': 1531, '弘治': 1554, '永禄': 1557, '元亀': 1569, '天正': 1572, '文禄': 1591, '慶長': 1595, '元和': 1614, '寛永': 1623, '正保': 1643, '慶安': 1647, '承応': 1651, '明暦': 1654, '万治': 1657, '寛文': 1660, '延宝': 1672, '天和': 1680, '貞享': 1683, '元禄': 1687, '宝永': 1703, '正徳': 1710, '享保': 1715, '元文': 1735, '寛保': 1740, '延享': 1743, '寛延': 1747, '宝暦': 1750, '明和': 1763, '安永': 1771, '天明': 1780, '寛政': 1788, '享和': 1800, '文化': 1803, '文政': 1817, '天保': 1829, '弘化': 1843, '嘉永': 1847, '安政': 1853, '万延': 1859, '文久': 1860, '元治': 1863, '慶応': 1864, '明治': 1867, '大正': 1911, '昭和': 1925, '平成': 1988, '令和': 2018}
offset_to_ja_era = {v: k for k, v in ja_era_to_offset.items()}


day_of_week_index = {
  'en': {
    0: {'word': 'monday', 'abbr': 'mon'},
    1: {'word': 'tuesday', 'abbr': 'tue'},
    2: {'word': 'wednesday', 'abbr': 'wed'},
    3: {'word': 'thursday', 'abbr': 'thu'},
    4: {'word': 'friday', 'abbr': 'fri'},
    5: {'word': 'saturday', 'abbr': 'sat'},
    6: {'word': 'sunday', 'abbr': 'sun'},

  },
  'ja': {
    0: {'word': '月曜日', 'abbr': '月曜'},
    1: {'word': '火曜日', 'abbr': '火曜'},
    2: {'word': '水曜日', 'abbr': '水曜'},
    3: {'word': '木曜日', 'abbr': '木曜'},
    4: {'word': '金曜日', 'abbr': '金曜'},
    5: {'word': '土曜日', 'abbr': '土曜'},
    6: {'word': '日曜日', 'abbr': '日曜'},


  },
  'ko': {
    0: {'word': '월요일', 'abbr': '월'},
    1: {'word': '화요일', 'abbr': '화'},
    2: {'word': '수요일', 'abbr': '수'},
    3: {'word': '목요일', 'abbr': '목'},
    4: {'word': '금요일', 'abbr': '금'},
    5: {'word': '토요일', 'abbr': '토'},
    6: {'word': '일요일', 'abbr': '일'},

  },

  'zh_hans': {
    0: {'word': '星期一', 'abbr': '周一'},
    1: {'word': '星期二', 'abbr': '周二'},
    2: {'word': '星期三', 'abbr': '周三'},
    3: {'word': '星期四', 'abbr': '周四'},
    4: {'word': '星期五', 'abbr': '周五'},
    5: {'word': '星期六', 'abbr': '周六'},
    6: {'word': '星期日', 'abbr': '周日'},

  },

  'zh_hant': {
    0: {'word': '星期一', 'abbr': '週一'},
    1: {'word': '星期二', 'abbr': '週二'},
    2: {'word': '星期三', 'abbr': '週三'},
    3: {'word': '星期四', 'abbr': '週四'},
    4: {'word': '星期五', 'abbr': '週五'},
    5: {'word': '星期六', 'abbr': '週六'},
    6: {'word': '星期日', 'abbr': '週日'},

  }
}
next_words = {
  'this': {
    'en': ["this "],
    'ja': ["今度の"],
    'ko': ["이번주"],
    'zh_hans': ["这个"],
    'zh_hant': ["这个"],
  },
  'next': {
    'en': ["next "],
    'ja': ["次の", "来週の"],
    'ko': ["다음"],
    'zh_hans': ["下周"],
    'zh_hant': ["下週"],
  },
  'last': {
    'en': ["last "],
    'ja': ["先週の"],
    'ko': ["지난"],
    'zh_hans': ["上个"],
    'zh_hant': ["上個"],
  }
}

fraction_words = {
  'en': {
    'half': 0.5,
    'third': 1./3.,
    'quarter': 0.25,
    'fifth': 1./5.,
    'sixth': 1./6.,
    'seventh': 1./7.,
    'eighth': 1./8.,
    'ninth': 1./9.,
    'tenth': .1
  }
}
