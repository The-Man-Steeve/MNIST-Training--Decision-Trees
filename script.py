import subprocess

c = 300
d = 100
mode = 'dt'

c_array = ['300', '500', '1000', '1500', '1800']
d_array = ['100', '1000', '5000']
mode_array = ['dt', 'bagging', 'rf', 'gb']

outfile = open(mode='w', file='complete_results.txt')

for i in range(len(c_array)):
    for j in range(len(d_array)):
        for k in range(len(mode_array)):
            #this was coded with the aid of AI
            test = subprocess.run(['python', 'dtclassifier.py', c_array[i], d_array[j], mode_array[k]],
                                  capture_output=True,
                                  text=True)
            output = test.stdout
            outfile.write(output)
outfile.close()