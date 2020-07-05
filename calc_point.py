import json

def calc_label():
    p_duration = 116 / 2
    pq_interval = 178 / 2
    qrs_duration = 100 / 2
    qt_interval = 394 / 2

    file_name = input("file name:")
    qrs_end = float(input("q_end:"))
    frequency = float(input("frequency:"))
    q_length = float(input("q length:"))

    result = {
        "p": [],
        "qrs": [],
        "t": [],
    }

    while qrs_end-qrs_duration+qt_interval < 5000:
        result["p"].append([qrs_end-qrs_duration-pq_interval, qrs_end-qrs_duration-pq_interval+p_duration])
        result["qrs"].append([qrs_end-qrs_duration, qrs_end])
        result["t"].append([qrs_end-qrs_duration+qt_interval-q_length, qrs_end-qrs_duration+qt_interval])
        qrs_end += frequency

    print(result)
    ret = json.dumps(result)
    with open("./label/"+file_name, "w") as f:
        f.write(ret)

def shift_label():
    file_name = input("file name:")
    for j in range(2, 6):    
        with open("./label/"+file_name + "_1.json", "r") as f:
            label = json.load(f)
        base = label["qrs"][0][1]
        new = float(input("new:"))
        shift = new - base
        for i in range(len(label["p"])):
            if label["p"][i][1] + shift < 5000:
                label["p"][i][0] += shift
                label["p"][i][1] += shift
        for i in range(len(label["qrs"])):
            if label["qrs"][i][1] + shift < 5000:
                label["qrs"][i][0] += shift
                label["qrs"][i][1] += shift
        for i in range(len(label["t"])):
            if label["t"][i][1] + shift < 5000:
                label["t"][i][0] += shift
                label["t"][i][1] += shift
        with open("./label/"+file_name + "_" + str(j) + ".json", "w") as f:
            f.write(json.dumps(label))
    

if __name__ == "__main__":
    shift_label()