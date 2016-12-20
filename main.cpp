
#include <iostream>
#include <math.h>
using namespace std;

//Network variables

const int PatternCount = 10;
const int InputNodes = 7;
const int HiddenNodes = 8;
const int OutputNodes = 4;
const float LearningRate = 0.4;
const float Momentum = 0.8;
const float InitialWeightMax = 0.5;
const float Success = 0.0004;

//training data - inputs and targets

const int Input[PatternCount][InputNodes] = {
        { 1, 1, 1, 1, 1, 1, 0 },  // 0
        { 0, 1, 1, 0, 0, 0, 0 },  // 1
        { 1, 1, 0, 1, 1, 0, 1 },  // 2
        { 1, 1, 1, 1, 0, 0, 1 },  // 3
        { 0, 1, 1, 0, 0, 1, 1 },  // 4
        { 1, 0, 1, 1, 0, 1, 1 },  // 5
        { 0, 0, 1, 1, 1, 1, 1 },  // 6
        { 1, 1, 1, 0, 0, 0, 0 },  // 7
        { 1, 1, 1, 1, 1, 1, 1 },  // 8
        { 1, 1, 1, 0, 0, 1, 1 }   // 9
};

const int Target[PatternCount][OutputNodes] = {
        { 0, 0, 0, 0 },
        { 0, 0, 0, 1 },
        { 0, 0, 1, 0 },
        { 0, 0, 1, 1 },
        { 0, 1, 0, 0 },
        { 0, 1, 0, 1 },
        { 0, 1, 1, 0 },
        { 0, 1, 1, 1 },
        { 1, 0, 0, 0 },
        { 1, 0, 0, 1 }
};

//variables for execution

int i, j, p, q, r;
int ReportEvery1000 = 1;
int RandomizedIndex[PatternCount];
long TrainingCycle;
float Rando;
float Error;
float Accum;

//setup the network and backpropogation arrays

float Hidden[HiddenNodes];
float Output[OutputNodes];

float HiddenWeights[InputNodes+1][HiddenNodes]; // +1 for the bias
float OutputWeights[HiddenNodes+1][OutputNodes];

float HiddenDelta[HiddenNodes]; //these are the sensitivities
float OutputDelta[OutputNodes];

float ChangeHiddenWeights[InputNodes+1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes+1][OutputNodes];

void InitialiseWeights() {

    for(i = 0; i < HiddenNodes; i++){
        for(j = 0; j <= InputNodes; j++){
            ChangeHiddenWeights[j][i] = 0.0; //set ChangeHiddenWeights to an array of zeros
            Rando = float(rand() % 100)/100;
            HiddenWeights[j][i] = 2.0 * (Rando = 0.5) * InitialWeightMax; //set the initial weights to random numbers
        }
    }

    for(i = 0; i < OutputNodes; i++){
        for(j = 0; j<= InputNodes; j++){
            ChangeOutputWeights[j][i] = 0.0;
            Rando = float(rand() % 100)/100;
            OutputWeights[j][i] = 2.0 * (Rando - 0.5) * InitialWeightMax;
        }
    }

    cout << "Inital Weights Randomised\n\n";

}

void toTerminal() {

    for(p = 0; p < PatternCount; p++) {

        cout << "\n\nTraining Pattern " << p; //output inputs and targets of training set

        cout << "\nInput: ";
        for(i = 0; i < InputNodes; i++) {
            cout << Input[p][i] << ", ";
        }

        cout << "\nTarget: ";
        for(i = 0; i < OutputNodes; i++) {
            cout << Target[p][i] << ", ";
        }


        for(i = 0; i < HiddenNodes; i++) { //calculate hidden layer activations of final run through
            Accum = HiddenWeights[InputNodes][i];

            for(j = 0; j < InputNodes; j++) {
                Accum += Input[p][j] * HiddenWeights[j][i];
            }

            Hidden[i] = 1.0/(1.0 + exp(-Accum));
        }

        for(i = 0; i < OutputNodes; i++) { //calculate output layer activations (and errors)
            Accum = OutputWeights[HiddenNodes][i];

            for(j = 0; j < HiddenNodes; j++) {
                Accum += Hidden[j] * OutputWeights[j][i];
            }

            Output[i] = round(1.0/(1.0 + exp(-Accum)));
        }

        cout << "\nOutput: ";
        for(i = 0; i < OutputNodes; i++) {
            cout << Output[i] << ", ";
        }

    }

}

int main(){

    for( p = 0; p < PatternCount; p++) {
        RandomizedIndex[p] = p; //creates a sequential array
    }

    InitialiseWeights();

    //BEGIN TRAINING

    for(TrainingCycle = 1; TrainingCycle < 1000000; TrainingCycle++) {

        for(p = 0 ; p < PatternCount; p++) { //Randomise order of training patterns - reduces risk of weights oscillating

            q = rand() % PatternCount;
            r = RandomizedIndex[p];
            RandomizedIndex[p] = RandomizedIndex[q];
            RandomizedIndex[q] = r;
        }

        Error = 0.0; // initialise error

        for(q = 0; q < PatternCount; q++) {
            p = RandomizedIndex[q]; // cycling through the randomised training patterns

            for(i = 0; i < HiddenNodes; i++){ // Computing hidden layer activations

                Accum  = HiddenWeights[InputNodes][i];

                for(j = 0; j < InputNodes; j++) {
                    Accum += Input[p][j] * HiddenWeights[j][i]; //sum of (inputs * hidden weights) for each hidden node
                }

                Hidden[i] = 1.0/(1.0 + exp(-Accum)); //activation for each hidden node with sigmoid function
            }

            for(i = 0; i < OutputNodes; i++) { // Computing output layer activations then errors

                Accum = OutputWeights[HiddenNodes][i];

                for(j = 0; j < HiddenNodes; j++) {
                    Accum += Hidden[j] * OutputWeights[j][i]; // sums (inputs * output weights) for each output node
                }

                Output[i] = 1.0/(1.0 + exp(-Accum)); //calculates output activation

                OutputDelta[i] = (Target[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]);
                Error += 0.5 * (Target[p][i] - Output[i]) * (Target[p][i] - Output[i]); 
            }

            //backpropogate the errors to the hidden layer

            for(i = 0; i < HiddenNodes; i++) {

                Accum = 0.0;

                for(j = 0; j < OutputNodes; j++) {
                    Accum += OutputWeights[i][j] * OutputDelta[j]; // product of output weights and output delta for each node - accum calculated for each hidden node
                }

                HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[j]); // calculates hidden delta
            }

            //update hidden weights

            for(i = 0; i < HiddenNodes; i++) {
                ChangeHiddenWeights[j][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i]; // determines the amount each weight will change by
                HiddenWeights[j][i] += ChangeHiddenWeights[j][i]; // implements change to weights

                for(j = 0; j < InputNodes; j++){
                    ChangeHiddenWeights[j][i] = LearningRate *  Input[p][j] * HiddenDelta[i] + Momentum * ChangeOutputWeights[j][i]; 
                    HiddenWeights[j][i] += ChangeHiddenWeights[j][i];
                }
            }

            //update output weights

            for(i = 0; i < OutputNodes; i++) {
                ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i];
                OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i];

                for(j = 0; j < HiddenNodes ; j++) {
                    ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i];
                    OutputWeights[j][i] += ChangeOutputWeights[j][i];
                }
            }

        } // ends cycling through each training pattern

        ReportEvery1000 = ReportEvery1000 - 1; //Reports progress every 1000 training cycles

        if(ReportEvery1000 == 0){
            cout << "Training Cycle: " << TrainingCycle;
            cout << "      Error: " << Error << "\n";

            //toTerminal();

            if(TrainingCycle == 1){ ReportEvery1000 = 999;}
            else { ReportEvery1000 = 1000; }
        }

        if(Error < Success) { break; } //if the training is complete, end the training loop.

    } //end of entire training loop

    cout << "\n~~~ Training Set Solved ~~~ \n\n";

    cout << "Training Cycle: " << TrainingCycle;
    cout << "\nError: " << Error ;

    toTerminal();

    cout << "\n\nComplete";
}























