import csv
import pickle


def main():
    session_swings = set()
    with open("./eligible_swings.csv") as file:
        csv_reader = csv.reader(file, delimiter=',')
        first = True
        keys = {}
        for line in csv_reader:
            if first:
                first = False
                for i in range(len(line)):
                    keys[line[i]] = i
                continue
            session_swings.add(line[keys["session_swing"]])
    #print(session_swings)

    final_data = {}
    count = 0
    keys_angles = []
    with open("./data/data/full_sig/joint_angles.csv") as file:
        csv_reader = csv.reader(file, delimiter=',')
        first = True
        keys = {}
        for line in csv_reader:
            #print(line)
            if count % 10000 == 0:
                print(count)
            count += 1
            if first:
                keys_angles = line
                first = False
                for i in range(len(line)):
                    keys[line[i]] = i
                continue
            sess = line[keys["session_swing"]]
            if sess in session_swings:
                if not sess in final_data:
                    final_data[sess] = [line]
                else:
                    for i in range(len(final_data[sess])):
                        if float(final_data[sess][i][keys["time"]]) > float(line[keys["time"]]):
                            final_data[sess].insert(i, line)
                            break
                        if i == len(final_data[sess]) - 1:
                            final_data[sess].append(line)
    keys_velos = []
    with open("./data/data/full_sig/joint_velos.csv") as file:
        csv_reader = csv.reader(file, delimiter=',')
        first = True
        keys = {}
        for line in csv_reader:
            #print(line)
            if count % 10000 == 0:
                print(count)
            count += 1
            if first:
                first = False
                keys_velos = line
                for i in range(len(line)):
                    keys[line[i]] = i
                continue
            sess = line[keys["session_swing"]]
            if sess in session_swings:
                for i in range(len(final_data[sess])):
                    if final_data[sess][i][keys["time"]] == line[keys["time"]]:
                        final_data[sess][i] += line
                        break
                    if i + 1 == len(final_data[sess]):
                        print("ERROR")
    final_data["keys"] = keys_angles + keys_velos
    with open('sorted_data.pkl', 'wb') as file:
        pickle.dump(final_data, file)
                    
                


if __name__ == "__main__":
    main()