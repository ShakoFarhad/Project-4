#include <iostream>
#include <cmath>
#include <armadillo>
#include <random>
#include <fstream>
#include <string>

using namespace std;
using namespace arma;


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


mat initialize(int rowSize=10, int columnSize=10, string fill="ones") {
    mat spinMatrix(rowSize, columnSize, fill::ones);

    if(fill.compare("randu") == 0) {
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

void writeToFile(string filename, double x, double y) {
    ofstream myfile;
    myfile.open(filename);

    myfile << to_string(x) + " \n";
    myfile << to_string(y) + " ";

    myfile.close();
}

vec metropolis(int nSpins=2, int monteCarloCycles=100000, double temperature=1.0, string fill="ordered", bool steadyState=false, bool makeProbDistHist=false) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    //Set up the uniform distribution for x \in [[0, 1]
    std::uniform_real_distribution<double> RandomNumberGenerator(0.0,1.0);

    //20 intervals which gives 19 histogram bars
    //This vector is for the prob dist histogram
    vec energyCount = zeros<vec>(20);

    //Initialize spinMatrix
    mat spinMatrix;
    if(fill.compare("ordered")==0) {
        spinMatrix = initialize(nSpins, nSpins, "ones");
    } else {
        spinMatrix = initialize(nSpins, nSpins, "randu");
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
                    if(cycles >= monteCarloCycles*0.1) {
                        //Steady state is (for now) set to be after 10% of the
                        //monte carlo calculations have been made
                        magneticMoment += 2.0*spinMatrix(ix,iy);
                        energy += (double) deltaE;
                        acceptedConfigurations++;

                        if(makeProbDistHist) {
                            //Count somehow

                        }
                    }
                } else {
                    //We calculate like normal if we do not care about steady state
                    magneticMoment += 2.0*spinMatrix(ix,iy);
                    energy += (double) deltaE;
                    acceptedConfigurations++;

                    if(makeProbDistHist) {
                        //count somehow
                    }
                }
            }
        }
        if(steadyState) {
            //We start calculating values only when we reach steady state
            if(cycles >= monteCarloCycles*0.1) {
                //Steady state is (for now) set to be after 10% of the
                //monte carlo calculations have been made
                //Update expectation values for local node
                relevantValues(0) += energy;
                relevantValues(1) += energy*energy;
                relevantValues(2) += magneticMoment;
                relevantValues(3) += magneticMoment*magneticMoment;
                relevantValues(4) += fabs(magneticMoment);
            }
        } else {
            //We calculate like normal if we do not care about steady state
            //Update expectation values for local node
            relevantValues(0) += energy;
            relevantValues(1) += energy*energy;
            relevantValues(2) += magneticMoment;
            relevantValues(3) += magneticMoment*magneticMoment;
            relevantValues(4) += fabs(magneticMoment);
        }
    }
    if(makeProbDistHist) {
        //NOT FINISHED
        //Writing the energies to file to create histogram
        ofstream filePD;
        string filenamePD = "ProbabilityDistribution"+fill+to_string((int)temperature)+".txt";
        filePD.open(filenamePD);
        for(int i = 0; i<20; i++) {
            filePD << to_string(i) + " ";
            filePD << to_string(energyCount(i)) + " \n";
        }
        //closing file
        filePD.close();
    }

    //Finding mean values
    relevantValues(0) /= monteCarloCycles; // E
    relevantValues(1) /= monteCarloCycles; // E^2
    relevantValues(2) /= monteCarloCycles; // M
    relevantValues(3) /= monteCarloCycles; // M^2
    relevantValues(4) /= monteCarloCycles; // |M|

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
    //NOT FINISHED
    int nSpins = 20; int monteCarloCycles = 100000;

    for(double temperature = 1.0; temperature < 2.5; temperature += 1.4) {
        vec relevantValues = metropolis(nSpins, monteCarloCycles, temperature, "Random", true, true);
    }
}


int main() {
    //expectedEnergyToFile(); //Finished and already ran it. Takes a long time
    //magneticMomentumToFile(); //Finished and already ran it. Takes a long time
    //acceptedConfigToFile(); //Finished and already ran it. Takes a long time

    return 0;
}
