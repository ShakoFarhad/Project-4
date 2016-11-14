#include <iostream>
#include <cmath>
#include <armadillo>
#include <random>
#include <fstream>
#include <string>
#include <cstdlib>
#include <chrono>

using namespace std;
using namespace arma;
using namespace std::chrono;

int pRow(int ix, int nSpins) {
    //Takes care of ix values on the row boundary
    int x = ix;

    if(ix == -1) {
        x = nSpins - 1;
    }
    if(ix == nSpins) {
        x = 0;
    }

    return x;
}

int pCol(int iy, int nSpins) {
    //Takes care of iy values on the column boundary
    int y = iy;

    if(iy == -1) {
        y = nSpins - 1;
    }
    if(iy == nSpins) {
        y = 0;
    }

    return y;
}


mat initialize(int rowSize=10, int columnSize=10, string fill="ordered") {
    mat spinMatrix(rowSize, columnSize, fill::ones);

    if(fill.compare("random") == 0) {
        mat spinRand(rowSize, columnSize, fill::randu);
        //spinMatrix is filled with random numbers between 0 and 1.
        //We must make these into 1 and -1

        for(int r = 0; r < rowSize; r++) {
            for(int c = 0; c < columnSize; c++) {
                if(spinRand(r,c) <=0.5) {
                    spinRand(r,c) = -1;
                } else {
                    spinRand(r,c) = 1;
                }
            }
        }
        spinMatrix = spinRand;
    }
    return spinMatrix;
}

void initializeEandM(int nSpins, mat spinMatrix, double& energy, double& magneticMoment) {
    //Initializing energy and magnetic moment

    for(int x = 0; x < nSpins; x++) {
        for(int y = 0; y < nSpins; y++) {
            magneticMoment += (double) spinMatrix(x,y);
            energy -= (double) spinMatrix(x,y)*(spinMatrix(x, pCol(y-1,nSpins))+
                                                spinMatrix(pRow(x-1,nSpins), y));
        }
    }
}

vec metropolis(int nSpins=2, int monteCarloCycles=100000, double temperature=1.0, string fill="ordered", bool steadyState=false, bool makeProbDistHist=false) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    //Set up the uniform distribution for x \in [[0, 1]
    std::uniform_real_distribution<double> RandomNumberGenerator(0.0,1.0);

    //nSpins*nSpins+1 intervals which gives nSpins*nSpins histogram bars
    //This vector is for the prob dist histogram
    vec energyCount = zeros<vec>(nSpins*nSpins+1);

    //Initialize spinMatrix
    mat spinMatrix;
    if(fill.compare("ordered")==0) {
        spinMatrix = initialize(nSpins, nSpins, "ordered");
    } else {
        spinMatrix = initialize(nSpins, nSpins, "random");
    }

    //Create vector to contain relevant values
    vec relevantValues = zeros<vec>(6);

    //Initialize energy, magnetization and accepted configurations counter
    double energy = 0.0; double magneticMoment = 0.0; int acceptedConfigurations = 0;
    initializeEandM(nSpins, spinMatrix, energy, magneticMoment);

    //setup array for possible energy changes
    vec energyDifference = zeros<mat>(17);

    //Precalculating the different e^{-\beta*E} values
    for(int de =-8; de <= 8; de+=4) {
        energyDifference(de+8) = exp(-de/temperature);
    }

    //Percentage of monte carlo cycles which we will ignore (reaching steady state after)
    double p = 0.1; //10%

    //Start Monte Carlo experiments
    int allSpins = nSpins*nSpins;
    for(int cycles = 1; cycles <= monteCarloCycles; cycles++){
        // The sweep over the lattice, looping over all spin sites
        for(int spins =0; spins < allSpins; spins++) {
            int ix = (int) (RandomNumberGenerator(gen)*nSpins);
            int iy = (int) (RandomNumberGenerator(gen)*nSpins);

            //Calculating energy difference
            int deltaE = 2*spinMatrix(ix,iy)*(spinMatrix(pRow(ix-1,nSpins),iy)+
                                              spinMatrix(pRow(ix+1,nSpins),iy)+
                                              spinMatrix(ix,pCol(iy-1,nSpins))+
                                              spinMatrix(ix,pCol(iy+1,nSpins)));

            if (RandomNumberGenerator(gen) <= energyDifference(deltaE+8)) {
                spinMatrix(ix,iy) *= -1.0;  //Flip one spin and accept new spin config

                if(steadyState) {
                    //We start calculating values only when we reach steady state
                    if(cycles >= monteCarloCycles*p) {
                        //Steady state is (for now) set to be after p% of the
                        //monte carlo calculations have been made
                        magneticMoment += 2.0*spinMatrix(ix,iy);
                        energy += (double) deltaE;
                        acceptedConfigurations++;
                    }
                } else {
                    //We calculate like normal if we do not care about steady state
                    magneticMoment += 2.0*spinMatrix(ix,iy);
                    energy += (double) deltaE;
                    acceptedConfigurations++;
                }
            }
        }
        if(steadyState) {
            //We start calculating values only when we reach steady state
            if(cycles >= monteCarloCycles*p) {
                //Steady state is (for now) set to be after p% of the
                //monte carlo calculations have been made
                //Update expectation values for local node
                relevantValues(0) += energy;
                relevantValues(1) += energy*energy;
                relevantValues(2) += magneticMoment;
                relevantValues(3) += magneticMoment*magneticMoment;
                relevantValues(4) += fabs(magneticMoment);

                //Counting the energies for use in the histogram plot
                energyCount((nSpins*nSpins*2 + (int)energy)/4)++;

            }
        } else {
            //We calculate like normal if we do not care about steady state
            //Update expectation values for local node
            relevantValues(0) += energy;
            relevantValues(1) += energy*energy;
            relevantValues(2) += magneticMoment;
            relevantValues(3) += magneticMoment*magneticMoment;
            relevantValues(4) += fabs(magneticMoment);

            //Counting the energies for use in the histogram plot
            energyCount((nSpins*nSpins*2 + (int)energy)/4)++;
        }
    }
    if(makeProbDistHist) {
        //NOT FINISHED
        //Writing the energies to file to create histogram
        ofstream filePD;
        string filenamePD = "ProbabilityDistribution"+fill+to_string((int)temperature)+".txt";
        filePD.open(filenamePD);
        for(int i = 0; i< nSpins*nSpins; i++) {
            filePD << to_string(i) + " ";
            //Dividing by monteCarloCycles to normalize and show the probability in the y-axis
            if(steadyState) {
                filePD << to_string(energyCount(i)/((double)monteCarloCycles*(1-p))) + " \n";
            } else {
                filePD << to_string(energyCount(i)/((double)monteCarloCycles)) + " \n";
            }
        }
        //closing file
        filePD.close();
    }
    if(steadyState) {
        //Finding mean values
        relevantValues(0) /= (monteCarloCycles*(1-p)*nSpins*nSpins); // E
        relevantValues(1) /= (monteCarloCycles*(1-p)*nSpins*nSpins); // E^2
        relevantValues(2) /= (monteCarloCycles*(1-p)*nSpins*nSpins); // M
        relevantValues(3) /= (monteCarloCycles*(1-p)*nSpins*nSpins); // M^2
        relevantValues(4) /= (monteCarloCycles*(1-p)*nSpins*nSpins); // |M|
    } else {
        //Finding mean values
        relevantValues(0) /= (monteCarloCycles*nSpins*nSpins); // E
        relevantValues(1) /= (monteCarloCycles*nSpins*nSpins); // E^2
        relevantValues(2) /= (monteCarloCycles*nSpins*nSpins); // M
        relevantValues(3) /= (monteCarloCycles*nSpins*nSpins); // M^2
        relevantValues(4) /= (monteCarloCycles*nSpins*nSpins); // |M|
    }
    //Adding relevant value
    relevantValues(5) = acceptedConfigurations;

    return relevantValues;
}

void expectedEnergyToFile() {
    //Writing E to file
    //Running for temp = 1.0 and temp = 2.4
    //And running for different monte carlo cycle numbers

    ofstream fileE;

    int nSpins = 20;

    string fill; //fill = "ordered" gives ordered spin orientation (1 in value everywhere)
                 //fill = "random" gives random spin orientation (1 and -1 randomly distributed)

    string filenameE;

    for(int i = 0; i < 2; i++) {
        //Running experiments with ordered spin orientation first, then random
        if(i==0) {
            fill = "ordered";
        } else {
            fill = "random";
        }
        for(double temperature = 1.0; temperature < 2.5; temperature += 1.4) {
            filenameE = "Energies"+fill+to_string((int)temperature)+".txt";
            fileE.open(filenameE);

            for(int monteCarloCycles = 100; monteCarloCycles < 1000000; monteCarloCycles *= 2) {
                vec relevantValues = metropolis(nSpins, monteCarloCycles, temperature, fill);

                fileE << to_string(monteCarloCycles) + " ";
                fileE << to_string(relevantValues(0)) + " \n";
            }
            fileE.close();
        }
    }
}

void magneticMomentumToFile() {
    //Writing |M| to file
    //Running for temp = 1.0 and temp = 2.4
    //And running for different monte carlo cycle numbers

    ofstream fileM;

    int nSpins = 20;

    string fill; //fill = "ordered" gives ordered spin orientation (1 in value everywhere)
                 //fill = "random" gives random spin orientation (1 and -1 randomly distributed)

    string filenameM;

    for(int i = 0; i < 2; i++) {
        //Running experiments with ordered spin orientation first, then random
        if(i==0) {
            fill = "ordered";
        } else {
            fill = "random";
        }
        for(double temperature = 1.0; temperature < 2.5; temperature += 1.4) {
            filenameM = "AbsMagneticMoments"+fill+to_string((int)temperature)+".txt";
            fileM.open(filenameM);

            for(int monteCarloCycles = 100; monteCarloCycles < 1000000; monteCarloCycles *= 2) {
                vec relevantValues = metropolis(nSpins, monteCarloCycles, temperature, fill);

                fileM << to_string(monteCarloCycles) + " ";
                fileM << to_string(relevantValues(4)) + " \n";
            }
            fileM.close();
        }
    }
}

void acceptedConfigToFile() {
    //Writing the number of accepted configurations to file
    //Running for temp = 1.0 and temp = 2.4
    //And running for different monte carlo cycle numbers

    ofstream fileAC;

    int nSpins = 20;

    string fill; //fill = "ordered" gives ordered spin orientation (1 in value everywhere)
                 //fill = "random" gives random spin orientation (1 and -1 randomly distributed)

    string filenameAC;

    for(int i = 0; i < 2; i++) {
        //Running experiments with ordered spin orientation first, then random
        if(i==0) {
            fill = "ordered";
        } else {
            fill = "random";
        }
        for(double temperature = 1.0; temperature < 2.5; temperature += 1.4) {
            filenameAC = "AcceptedConfigurations"+fill+to_string((int)temperature)+".txt";
            fileAC.open(filenameAC);

            for(int monteCarloCycles = 100; monteCarloCycles < 1000000; monteCarloCycles *= 2) {
                vec relevantValues = metropolis(nSpins, monteCarloCycles, temperature, fill);

                fileAC << to_string(monteCarloCycles) + " ";
                fileAC << to_string(relevantValues(5)) + " \n";
            }
            fileAC.close();
        }
    }
}


void probDistToFile() {
    int nSpins = 20; int monteCarloCycles = 100000;

    for(double temperature = 1.0; temperature < 2.5; temperature += 1.4) {
        vec relevantValues = metropolis(nSpins, monteCarloCycles, temperature, "Random", true, true);
    }
}

void phaseTransitionNotParallell(double tempStart, double tempEnd, double dt) {
    int nSpins[] = {40,60,100,140};
    int monteCarloCycles = 100000;

    double heatSpecific;
    double susceptibility;
    double varians;

    for(int i = 0; i < 4; i++) {
        for(double T = tempStart; T <= tempEnd; T += dt) {
            vec relevantValues = metropolis(nSpins[i], monteCarloCycles, T, "Random");
            varians = relevantValues(1) - relevantValues(0)*relevantValues(0);
            heatSpecific = (1/(T*T))*varians;
            susceptibility = (relevantValues(3)-relevantValues(2)*relevantValues(2))/T;
        }
    }
}

void phaseTransitionParallell(double tempStart, double tempEnd, double dt, int worker, bool writeToFile=false) {
    int nSpins[] = {40,60,100,140};
    int monteCarloCycles = 10000000;
    double numberOfJobs = ((tempEnd - tempStart)/dt + 1)/6.0;
    double T[(int)((tempEnd - tempStart)/dt)];

    for(int i = 0; i <= (tempEnd - tempStart)/dt; i++) {
        T[i] = tempStart + i*dt;
    }

    ofstream filePTE;
    string filenamePTE;

    ofstream filePTM;
    string filenamePTM;

    ofstream filePTCV;
    string filenamePTCV;

    ofstream filePTX;
    string filenamePTX;

    double heatSpecific;
    double susceptibility;
    double varians;

    for(int i = 0; i < 4; i++) {
        for(int k = (worker-1)*numberOfJobs; k < worker*numberOfJobs; k++) {
            vec relevantValues = metropolis(nSpins[i], monteCarloCycles, T[k], "Random");

            varians = relevantValues(1) - relevantValues(0)*relevantValues(0);
            heatSpecific = (1/(T[k]*T[k]))*varians;
            susceptibility = (relevantValues(3)-relevantValues(2)*relevantValues(2))/T[k];

            if(writeToFile) {
                filenamePTE = "ExpectedEnergyAsFunctionOfTemperature"+to_string(nSpins[i])+to_string(worker)+".txt";
                filePTE.open(filenamePTE);

                filePTE << to_string(T[k]) + " ";
                filePTE << to_string(relevantValues(0)) + " \n";



                filenamePTM = "AbsoluteValueMagneticMomentumAsFunctionOfTemperature"+to_string(nSpins[i])+to_string(worker)+".txt";
                filePTM.open(filenamePTM);

                filePTM << to_string(T[k]) + " ";
                filePTM << to_string(relevantValues(4)) + " \n";



                filenamePTCV = "HeatSpecificAsfunctionOfTemperature"+to_string(nSpins[i])+to_string(worker)+".txt";
                filePTCV.open(filenamePTCV);

                filePTCV << to_string(T[k]) + " ";
                filePTCV << to_string(heatSpecific) + " \n";



                filenamePTX = "SuceptibilityAsFunctionOfTemperature"+to_string(nSpins[i])+to_string(worker)+".txt";
                filePTX.open(filenamePTX);

                filePTX << to_string(T[k]) + " ";
                filePTX << to_string(susceptibility) + " \n";
            }
        }
        if(writeToFile) {
            filePTE.close();
            filePTM.close();
            filePTCV.close();
            filePTX.close();
        }
    }
}

int main(int argc, char* argv[]) {
    //expectedEnergyToFile(); //Finished and already ran it. Takes a long time. See github for txt file
    //magneticMomentumToFile(); //Finished and already ran it. Takes a long time. See github for txt file
    //acceptedConfigToFile(); //Finished and already ran it. Takes a long time. See github for txt file
    //probDistToFile(); //Finished and already ran it. See github for txt file


    if(argc < 2) {
        //This means that we will not use parallellization
        double tempStart = 2.0; double tempEnd = 2.3; double dt = (tempEnd - tempStart)/(6.0*2.0-1.0);
        double elapsedTime;

        high_resolution_clock::time_point time1 = high_resolution_clock::now();

        phaseTransitionNotParallell(tempStart, tempEnd, dt);

        high_resolution_clock::time_point time2 = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>( time2 - time1 ).count();

        elapsedTime = duration;

        cout << "Program finished the task in " << elapsedTime << " microseconds." << endl;
        cout << endl;

    } else if(argc == 2) {
        //We will use parallellization
        string str = string(argv[1]);

        if(str.compare("help") == 0 || str.compare("Help") == 0 || str.compare("HELP") == 0 ) {
            cout << "This program will divide the workload in 6 parts and" << endl;
            cout << "complete one part for each call." << endl;
            cout << "To use the program with parallellization follow the example below" << endl;
            cout << "./Project4 1" << endl;
            cout << "./Project4 2" << endl;
            cout << "./Project4 3" << endl;
            cout << "./Project4 4" << endl;
            cout << "./Project4 5" << endl;
            cout << "./Project4 6" << endl;
            cout << endl;
            cout << "We have here ran the same program 6 times and" << endl;
            cout << "therefore have now run the program for all parts." << endl;
            cout << endl;
            cout << "To write values to file, add 'write' as second argument." << endl;
            cout << endl;
            cout << "This needs to be written in the terminal while being" << endl;
            cout << "in the folder where Project4.x is." << endl;

        } else {
            double tempStart = 2.0; double tempEnd = 2.3; double dt = (tempEnd - tempStart)/(6.0*2.0-1.0);
            int worker = atoi(argv[1]);

            double elapsedTime;

            if(worker <= 6) {
                high_resolution_clock::time_point time1 = high_resolution_clock::now();

                phaseTransitionParallell(tempStart, tempEnd, dt, worker);

                high_resolution_clock::time_point time2 = high_resolution_clock::now();
                auto duration = duration_cast<microseconds>( time2 - time1 ).count();

                elapsedTime = duration;

                cout << "Worker number " << worker << " finished in " << elapsedTime << " microseconds." << endl;
                cout << endl;
            } else {
                cout << "Only accepting 6 workers" << endl;
            }
        }
    } else if(argc > 2) {
        double tempStart = 2.0; double tempEnd = 2.3; double dt = (tempEnd - tempStart)/(6.0*2.0-1.0);

        string str2 = string(argv[2]);

        if(str2.compare("write") == 0 || str2.compare("Write") == 0 || str2.compare("WRITE") == 0) {
            int worker = atoi(argv[1]);
            if(worker <= 6) {
                phaseTransitionParallell(tempStart, tempEnd, dt, worker, true);

                cout << "Worker number " << worker << " has finished the job." << endl;
                cout << endl;
            } else {
                cout << "Only accepting 6 workers" << endl;
            }
        } else {
            cout << "Wrong arguments. Run the program with 'help' as argument for guidelines." << endl;
        }
    }



    return 0;
}
