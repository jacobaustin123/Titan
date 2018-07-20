import bindloch

#Set up simulation 

sim =  bindloch.Sim()

m1 = bindloch.Mass() #Throws memory error
m1 = sim.createMass() #Throws memory error
runtime = 10 #How long the simulation runs for
print(sim.running) #Should print 0
sim.start(runtime) #Runs for 10 virtual seconds when the precompiler flag for constraints is not set.



