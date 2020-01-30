def ISO6346(temp):
text = []
meaning = []
category = ['U', 'Z', 'J']
info = ['NULL', 'NULL', 'NULL', 'NULL', 'NULL']

for char in temp:
    if temp[-1] in category:
        bool = "owner"
    if char.isalpha() is False:
        bool = "typeCode"
        break
if len(temp) != 4 or temp == "TARE":
    bool = "unknown"
if len(temp) == 6 or len(temp) == 7:
    for char in temp:
        bool = "serial"
        if char.isnumeric() is False:
            bool = "unknown"
            break
meaning.append(bool)
text.append(temp)