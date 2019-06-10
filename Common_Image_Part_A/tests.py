from FOTFlib import main

#thresh=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
thresh=[0.25,0.35,0.45]
dbscan_epsilon=[10,15,30]

for threshold in thresh:
    for dbscan in dbscan_epsilon:
        print("$$$$$$$$$$$$ test with tresh: "+str(threshold)+" and dbscan: "+str(dbscan))
        main("source/3.6.19/1/server", "source/3.6.19/1/client/10.jpg", "source/3.6.19/1/output1/" + str(threshold)+"/"+str(dbscan),threshold,dbscan)
        print("$@$@$@$@$@$@ test: ")
        main("source/3.6.19/1/server", "source/3.6.19/1/client/13.jpg", "source/3.6.19/1/output2/"+ str(threshold)+"/"+str(dbscan),threshold,dbscan)
        print("$@$@$@$@$@$@ test: source/3.6.19/1/client/13.jpg")
        main("source/3.6.19/2/server", "source/3.6.19/2/client/test_1/77.jpg", "source/3.6.19/2/output1/"+ str(threshold)+"/"+str(dbscan),threshold,dbscan)
        print("$@$@$@$@$@$@ test: source/3.6.19/2/client/test_1/77.jpg")
        main("source/3.6.19/2/server", "source/3.6.19/2/client/73.jpg", "source/3.6.19/2/output2/"+ str(threshold)+"/"+str(dbscan),threshold,dbscan)
        print("$@$@$@$@$@$@ test: source/3.6.19/2/client/73.jpg")
        main("source/3.6.19/3/server", "source/3.6.19/3/client/test_1/19.jpg", "source/3.6.19/3/output1/"+ str(threshold)+"/"+str(dbscan),threshold,dbscan)
        print("$@$@$@$@$@$@ test: source/3.6.19/3/client/test_1/19.jpg")
        main("source/3.6.19/3/server", "source/3.6.19/3/client/90.jpg", "source/3.6.19/3/output2/"+ str(threshold)+"/"+str(dbscan),threshold,dbscan)
        print("$@$@$@$@$@$@ test: source/3.6.19/3/client/90.jpg")
        main("source/3.6.19/3/server", "source/3.6.19/3/client/277.jpg", "source/3.6.19/3/output3/"+ str(threshold)+"/"+str(dbscan),threshold,dbscan)
        print("$@$@$@$@$@$@ test: source/3.6.19/3/client/277.jpg")
        main("source/3.6.19/4/server", "source/3.6.19/4/client/test_1/121.jpg", "source/3.6.19/4/output1/"+ str(threshold)+"/"+str(dbscan),threshold,dbscan)
        print("$@$@$@$@$@$@ test: source/3.6.19/4/client/test_1/121.jpg")
        main("source/3.6.19/4/server", "source/3.6.19/4/client/115.jpg", "source/3.6.19/4/output2/"+ str(threshold)+"/"+str(dbscan),threshold,dbscan)
        print("$@$@$@$@$@$@ test: source/3.6.19/4/client/115.jpg")
        main("source/3.6.19/4/server", "source/3.6.19/4/client/183.jpg", "source/3.6.19/4/output3/"+ str(threshold)+"/"+str(dbscan),threshold,dbscan)
        print("$@$@$@$@$@$@ test: source/3.6.19/4/client/183.jpg")



'''for j in range(1,6):
    print("round: "+str(j))
    main("source/3.6.19/1/server","source/3.6.19/1/client/10.jpg","source/3.6.19/1/output1/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/1/server","source/3.6.19/1/client/10.jpg","source/3.6.19/1/output1/"+str(thresh[j]),thresh[j])

for j in range(1,6):
    print("round: "+str(j))
    main("source/3.6.19/1/server","source/3.6.19/1/client/13.jpg","source/3.6.19/1/output2/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/1/server","source/3.6.19/1/client/13.jpg","source/3.6.19/1/output2/"+str(thresh[j]),thresh[j])


for j in range(1,6):
    print("round: "+str(j))
    main("source/3.6.19/2/server","source/3.6.19/2/client/test_1/77.jpg","source/3.6.19/2/output1/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/2/server","source/3.6.19/2/client/test_1/77.jpg","source/3.6.19/2/output1/"+str(thresh[j]),thresh[j])

for j in range(1,6):
    print("round: "+str(j))
    main("source/3.6.19/2/server","source/3.6.19/2/client/73.jpg","source/3.6.19/2/output2/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/2/server","source/3.6.19/2/client/73.jpg","source/3.6.19/2/output2/"+str(thresh[j]),thresh[j])

for j in range(1,6):
    print("round: "+str(j))
    main("source/3.6.19/3/server","source/3.6.19/3/client/test_1/19.jpg","source/3.6.19/3/output1/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/3/server","source/3.6.19/3/client/test_1/19.jpg","source/3.6.19/3/output1/"+str(thresh[j]),thresh[j])

for j in range(1,6):
    print("round: "+str(j))
    main("source/3.6.19/3/server","source/3.6.19/3/client/90.jpg","source/3.6.19/3/output2/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/3/server","source/3.6.19/3/client/90.jpg","source/3.6.19/3/output2/"+str(thresh[j]),thresh[j])

for j in range(1,6):
    print("round: "+str(j))
    main("source/3.6.19/3/server","source/3.6.19/3/client/277.jpg","source/3.6.19/3/output3/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/3/server","source/3.6.19/3/client/277.jpg","source/3.6.19/3/output3/"+str(thresh[j]),thresh[j])

for j in range(1,6):
    print("round: "+str(j))
    main("source/3.6.19/4/server","source/3.6.19/4/client/test_1/121.jpg","source/3.6.19/4/output1/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/4/server","source/3.6.19/4/client/test_1/121.jpg","source/3.6.19/4/output1/"+str(thresh[j]),thresh[j])

for j in range(1,6):
    print("round: "+str(j))
    c+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/4/server","source/3.6.19/4/client/115.jpg","source/3.6.19/4/output2/"+str(thresh[j]),thresh[j])

for j in range(1,6):
    print("round: "+str(j))
    main("source/3.6.19/4/server","source/3.6.19/4/client/183.jpg","source/3.6.19/4/output3/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/4/server","source/3.6.19/4/client/183.jpg","source/3.6.19/4/output3/"+str(thresh[j]),thresh[j])

'''
#tests befor objects
'''
thresh=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
for j in range(0,11):
    print("round: "+str(j))
    main("source/3.6.19/1/server","source/3.6.19/1/client/97.jpg","source/3.6.19/1/output/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/1/server","source/3.6.19/1/client/97.jpg","source/3.6.19/1/output/"+str(thresh[j]),thresh[j])


for j in range(0,11):
    print("round: "+str(j))
    main("source/3.6.19/2/server","source/3.6.19/2/client/186.jpg","source/3.6.19/2/output/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/2/server","source/3.6.19/2/client/186.jpg","source/3.6.19/2/output/"+str(thresh[j]),thresh[j])

for j in range(0,11):
    print("round: "+str(j))
    main("source/3.6.19/3/server","source/3.6.19/3/client/120.jpg","source/3.6.19/3/output/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/3/server","source/3.6.19/3/client/120.jpg","source/3.6.19/3/output/"+str(thresh[j]),thresh[j])

for j in range(0,11):
    print("round: "+str(j))
    main("source/3.6.19/4/server","source/3.6.19/4/client/6.jpg","source/3.6.19/4/output/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/4/server","source/3.6.19/4/client/6.jpg","source/3.6.19/4/output/"+str(thresh[j]),thresh[j])

for j in range(0,11):
    print("round: "+str(j))
    main("source/3.6.19/5/server","source/3.6.19/5/client/77.jpg","source/3.6.19/5/output/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/5/server","source/3.6.19/5/client/77.jpg","source/3.6.19/5/output/"+str(thresh[j]),thresh[j])

for j in range(0,11):
    print("round: "+str(j))
    main("source/3.6.19/6/server","source/3.6.19/6/client/19.jpg","source/3.6.19/6/output/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/6/server","source/3.6.19/6/client/19.jpg","source/3.6.19/6/output/"+str(thresh[j]),thresh[j])

for j in range(0,11):
    print("round: "+str(j))
    main("source/3.6.19/7/server","source/3.6.19/7/client/121.jpg","source/3.6.19/7/output/"+str(j/10),j/10)
    if j<10:
        main("source/3.6.19/7/server","source/3.6.19/7/client/121.jpg","source/3.6.19/7/output/"+str(thresh[j]),thresh[j])

'''
