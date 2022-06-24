import os
import random
import sys
import optparse
import traci
import 
import matplotlib.pyplot as plt
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

if __name__ == "__main__":
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    traci.start([sumoBinary, '-c', 'C:\\Users\\于\\Desktop\\sumo-test\\test.sumocfg'])
    Speed_step = [[],[],[],[],[]]
    Position_step = [[],[],[],[],[]]
    actual_dis = []
    target_dis = []
    for step in range(0,210):
         # if step>5:
            # traci.vehicle.setSpeed("a11.0",10)
            #print(traci.vehicle.getSpeed('a11.0'))
         if step==9:
            traci.vehicle.setSpeed('a11.0', random.uniform(8,10))
         if step>=10:
            traci.vehicle.setSpeed('a11.0', random.uniform(8,10))
            v1 = traci.vehicle.getSpeed('a11.0')
            v2 = traci.vehicle.getSpeed('a11.1')
            v3 = traci.vehicle.getSpeed('a11.2')
            v4 = traci.vehicle.getSpeed('a11.3')
            v5 = traci.vehicle.getSpeed('a11.4')
            p1 = traci.vehicle.getPosition('a11.0')[0]
            p2 = traci.vehicle.getPosition('a11.1')[0]
            p3 = traci.vehicle.getPosition('a11.2')[0]
            p4 = traci.vehicle.getPosition('a11.3')[0]
            p5 = traci.vehicle.getPosition('a11.4')[0]
            Speed_step[0].append(v1)
            Speed_step[1].append(v2)
            Speed_step[2].append(v3)
            Speed_step[3].append(v4)
            Speed_step[4].append(v5)
            Position_step[0].append(p1)
            Position_step[1].append(p2)
            Position_step[2].append(p3)
            Position_step[3].append(p4)
            Position_step[4].append(p5)
            actual_dis.append(p1-p2)
            target_dis.append(15)
            print(traci.vehicle.getSpeed('a11.0'), traci.vehicle.getSpeed('a11.1'),traci.vehicle.getSpeed('a11.2'), traci.vehicle.getSpeed('a11.3'),traci.vehicle.getSpeed('a11.4'))
            print(p1,p2,p3,p4,p5)
          #traci.vehicle.setSpeed('a11.0',10)
          #print(traci.vehicle.getSpeed('a11.0'))
         #print(traci.vehicle.getIDList())
#        print(traci.edge.getIDList())
#        print(traci.inductionloop.getVehicleData('abcd'))
         traci.simulationStep()
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 30})
    plt.plot(Speed_step[0])
    plt.plot(Speed_step[1])
    plt.plot(Speed_step[2])
    plt.plot(Speed_step[3])
    plt.plot(Speed_step[4])
    plt.legend(["PL", "PM1","PM2","PM3","PM4"], loc="upper left")
    plt.ylabel("Speed",fontsize = 30)
    plt.xlabel("时间/s",fontsize = 30)
    plt.show()

    plt.plot(Speed_step[0])
    plt.plot(Speed_step[1])

    plt.legend(["PL", "PM1"], loc="upper left")
    plt.ylabel("Speed", fontsize=30)
    plt.xlabel("时间/s", fontsize=30)
    plt.show()

    plt.plot(Position_step[0])
    plt.plot(Position_step[1])
    plt.plot(Position_step[2])
    plt.plot(Position_step[3])
    plt.plot(Position_step[4])
    plt.legend(["PL", "PM1", "PM2", "PM3", "PM4"], loc="upper left")
    plt.ylabel("Position",fontsize = 30)
    plt.xlabel("时间/s",fontsize = 30)
    plt.show()

    plt.plot(Position_step[0])
    plt.plot(Position_step[1])
    plt.legend(["PL", "PM1"], loc="upper left")
    plt.ylabel("Position",fontsize = 30)
    plt.xlabel("时间/s",fontsize = 30)
    plt.show()

    plt.plot(target_dis)
    plt.plot(actual_dis)
    plt.legend(["期望车距", "实际车距"], loc="upper left")
    plt.ylabel("车间距/m", fontsize=30)
    plt.xlabel("时间/s", fontsize=30)
    plt.show()
    traci.close()
