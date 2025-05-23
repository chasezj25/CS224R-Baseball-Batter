import csv
"""
Filter out the left handed swings for now.

KEYS
'session_swing',
 'session_mass_lbs', 
 'session_height_in', 
 'athlete_age', 
 'highest_playing_level', 
 'hitter_side', 
 'bat_weight_oz', 
 'bat_length_in', 
 'bat_speed_mph_max_x', 
 'blast_bat_speed_mph_x', 
 'exit_velo_mph_x', 
 'user', 
 'session'
"""

filter_out = {
    "hitter_side": "L"
}
def main():
    with open("../data/data/metadata.csv") as file:
        csv_reader = csv.reader(file, delimiter=',')
        first = True
        keys = {}
        finals = []
        for row in  csv_reader:
            #print(row)
            if first:
                first = False
                finals.append(row)
                for i in range(len(row)):
                    keys[row[i]] = i
            else:
                add = True
                for key in filter_out.keys():
                    if row[keys[key]] == filter_out[key]:
                        add = False
                        break
                if add:
                    finals.append(row)
        
    with open("../eligible_swings.csv", "w") as file:
        csv_writer = csv.writer(file, delimiter=',')
        for line in finals:
            csv_writer.writerow(line)



            
            


if __name__ == "__main__":
    main()