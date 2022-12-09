
path = '/Users/helencho/Desktop/senior/cs243/automatic-bassoon/plots/docker_data/hc_finegrained_lr_bs1024_shufflenetv2/cifar100+shufflenetv2+bs1024+adam+lr0.001+tm0.6+me100+x100+eta0.5+beta2.0+2022120817401670539224/master.log'

# Go through line by line and write new file
with open(path, 'r') as log:
    line = log.readline()
    filename = ''
    f = None
    writing = False
    while line:
        if 'Launching' in line:
            if writing:
                f.close()
            lr = float(line[line.index('LR') + len('LR: '):])
            # Write this and every other line after to a new file
            filename = f'lr_{lr}_adam_bs1024.txt'
            f = open(filename, 'w')
            f.write(line)
            writing = True
        else:
            if writing:
                f.write(line)
        line = log.readline()
    
    log.close()

