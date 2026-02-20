#ifndef MSCKF_CORE_LSTM_PREDICTOR_HPP
#define MSCKF_CORE_LSTM_PREDICTOR_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../hal/imu_types.hpp"
#include <cstring>

namespace msckf {

constexpr uint32 LSTM_INPUT_SIZE = 6;
constexpr uint32 LSTM_HIDDEN_SIZE = 64;
constexpr uint32 LSTM_NUM_LAYERS = 2;
constexpr uint32 LSTM_SEQUENCE_LENGTH = 10;
constexpr uint32 LSTM_WEIGHTS_SIZE = 4096;

struct LSTMCell {
    Matrix<LSTM_HIDDEN_SIZE, LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE, float64> W_ii;
    Matrix<LSTM_HIDDEN_SIZE, LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE, float64> W_if;
    Matrix<LSTM_HIDDEN_SIZE, LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE, float64> W_ig;
    Matrix<LSTM_HIDDEN_SIZE, LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE, float64> W_io;
    
    Vector<LSTM_HIDDEN_SIZE, float64> b_ii;
    Vector<LSTM_HIDDEN_SIZE, float64> b_if;
    Vector<LSTM_HIDDEN_SIZE, float64> b_ig;
    Vector<LSTM_HIDDEN_SIZE, float64> b_io;
    
    Vector<LSTM_HIDDEN_SIZE, float64> h;
    Vector<LSTM_HIDDEN_SIZE, float64> c;
    
    LSTMCell() {
        h = Vector<LSTM_HIDDEN_SIZE, float64>();
        c = Vector<LSTM_HIDDEN_SIZE, float64>();
    }
    
    void forward(const Vector<LSTM_INPUT_SIZE, float64>& input) {
        Vector<LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE, float64> combined;
        for (uint32 i = 0; i < LSTM_INPUT_SIZE; ++i) {
            combined[i] = input[i];
        }
        for (uint32 i = 0; i < LSTM_HIDDEN_SIZE; ++i) {
            combined[LSTM_INPUT_SIZE + i] = h[i];
        }
        
        Vector<LSTM_HIDDEN_SIZE, float64> i_gate = sigmoid(W_ii * combined + b_ii);
        Vector<LSTM_HIDDEN_SIZE, float64> f_gate = sigmoid(W_if * combined + b_if);
        Vector<LSTM_HIDDEN_SIZE, float64> g_gate = tanh(W_ig * combined + b_ig);
        Vector<LSTM_HIDDEN_SIZE, float64> o_gate = sigmoid(W_io * combined + b_io);
        
        c = f_gate * c + i_gate * g_gate;
        h = o_gate * tanh(c);
    }
    
    Vector<LSTM_HIDDEN_SIZE, float64> sigmoid(const Vector<LSTM_HIDDEN_SIZE, float64>& x) {
        Vector<LSTM_HIDDEN_SIZE, float64> result;
        for (uint32 i = 0; i < LSTM_HIDDEN_SIZE; ++i) {
            result[i] = 1.0 / (1.0 + exp(-x[i]));
        }
        return result;
    }
    
    Vector<LSTM_HIDDEN_SIZE, float64> tanh(const Vector<LSTM_HIDDEN_SIZE, float64>& x) {
        Vector<LSTM_HIDDEN_SIZE, float64> result;
        for (uint32 i = 0; i < LSTM_HIDDEN_SIZE; ++i) {
            result[i] = ::tanh(x[i]);
        }
        return result;
    }
    
    Vector<LSTM_HIDDEN_SIZE, float64> operator*(const Vector<LSTM_HIDDEN_SIZE, float64>& a,
                                                  const Vector<LSTM_HIDDEN_SIZE, float64>& b) {
        Vector<LSTM_HIDDEN_SIZE, float64> result;
        for (uint32 i = 0; i < LSTM_HIDDEN_SIZE; ++i) {
            result[i] = a[i] * b[i];
        }
        return result;
    }
    
    Vector<LSTM_HIDDEN_SIZE, float64> operator+(const Vector<LSTM_HIDDEN_SIZE, float64>& a,
                                                  const Vector<LSTM_HIDDEN_SIZE, float64>& b) {
        Vector<LSTM_HIDDEN_SIZE, float64> result;
        for (uint32 i = 0; i < LSTM_HIDDEN_SIZE; ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    }
    
    Vector<LSTM_HIDDEN_SIZE, float64> W_ii_mul(const Vector<LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE, float64>& x) {
        Vector<LSTM_HIDDEN_SIZE, float64> result;
        for (uint32 i = 0; i < LSTM_HIDDEN_SIZE; ++i) {
            float64 sum = 0;
            for (uint32 j = 0; j < LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE; ++j) {
                sum += W_ii(i, j) * x[j];
            }
            result[i] = sum;
        }
        return result;
    }
    
    Vector<LSTM_HIDDEN_SIZE, float64> W_if_mul(const Vector<LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE, float64>& x) {
        Vector<LSTM_HIDDEN_SIZE, float64> result;
        for (uint32 i = 0; i < LSTM_HIDDEN_SIZE; ++i) {
            float64 sum = 0;
            for (uint32 j = 0; j < LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE; ++j) {
                sum += W_if(i, j) * x[j];
            }
            result[i] = sum;
        }
        return result;
    }
    
    Vector<LSTM_HIDDEN_SIZE, float64> W_ig_mul(const Vector<LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE, float64>& x) {
        Vector<LSTM_HIDDEN_SIZE, float64> result;
        for (uint32 i = 0; i < LSTM_HIDDEN_SIZE; ++i) {
            float64 sum = 0;
            for (uint32 j = 0; j < LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE; ++j) {
                sum += W_ig(i, j) * x[j];
            }
            result[i] = sum;
        }
        return result;
    }
    
    Vector<LSTM_HIDDEN_SIZE, float64> W_io_mul(const Vector<LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE, float64>& x) {
        Vector<LSTM_HIDDEN_SIZE, float64> result;
        for (uint32 i = 0; i < LSTM_HIDDEN_SIZE; ++i) {
            float64 sum = 0;
            for (uint32 j = 0; j < LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE; ++j) {
                sum += W_io(i, j) * x[j];
            }
            result[i] = sum;
        }
        return result;
    }
    
    void forward_optimized(const Vector<LSTM_INPUT_SIZE, float64>& input) {
        Vector<LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE, float64> combined;
        for (uint32 i = 0; i < LSTM_INPUT_SIZE; ++i) {
            combined[i] = input[i];
        }
        for (uint32 i = 0; i < LSTM_HIDDEN_SIZE; ++i) {
            combined[LSTM_INPUT_SIZE + i] = h[i];
        }
        
        Vector<LSTM_HIDDEN_SIZE, float64> i_gate = sigmoid(W_ii_mul(combined) + b_ii);
        Vector<LSTM_HIDDEN_SIZE, float64> f_gate = sigmoid(W_if_mul(combined) + b_if);
        Vector<LSTM_HIDDEN_SIZE, float64> g_gate = tanh(W_ig_mul(combined) + b_ig);
        Vector<LSTM_HIDDEN_SIZE, float64> o_gate = sigmoid(W_io_mul(combined) + b_io);
        
        c = f_gate * c + i_gate * g_gate;
        h = o_gate * tanh(c);
    }
};

class LSTMPredictor {
public:
    struct Config {
        uint32 inputSize;
        uint32 hiddenSize;
        uint32 numLayers;
        uint32 sequenceLength;
        float64 learningRate;
        float64 momentum;
        float64 weightDecay;
        bool useOnlineLearning;
        bool useQuantization;
        
        Config() 
            : inputSize(LSTM_INPUT_SIZE)
            , hiddenSize(LSTM_HIDDEN_SIZE)
            , numLayers(LSTM_NUM_LAYERS)
            , sequenceLength(LSTM_SEQUENCE_LENGTH)
            , learningRate(0.001)
            , momentum(0.9)
            , weightDecay(0.0001)
            , useOnlineLearning(true)
            , useQuantization(true) {}
    };

private:
    Config config_;
    
    LSTMCell layers_[LSTM_NUM_LAYERS];
    
    Matrix<6, LSTM_HIDDEN_SIZE, float64> outputLayer_;
    Vector<6, float64> outputBias_;
    
    Matrix<6, 6, float64> predictionCovariance_;
    
    IMUData imuBuffer_[LSTM_SEQUENCE_LENGTH];
    uint32 bufferHead_;
    uint32 bufferSize_;
    
    Vector3d predictedAccelBias_;
    Vector3d predictedGyroBias_;
    
    float64 weights_[LSTM_WEIGHTS_SIZE];
    int16_t quantizedWeights_[LSTM_WEIGHTS_SIZE];
    
    float64* gradients_;
    float64* velocity_;
    
    uint32 totalPredictions_;
    float64 predictionErrorSum_;
    float64 lastPredictionError_;

public:
    LSTMPredictor() 
        : bufferHead_(0)
        , bufferSize_(0)
        , totalPredictions_(0)
        , predictionErrorSum_(0)
        , lastPredictionError_(0) {
        
        predictedAccelBias_ = Vector3d({0, 0, 0});
        predictedGyroBias_ = Vector3d({0, 0, 0});
        predictionCovariance_ = Matrix<6, 6, float64>::identity();
        
        gradients_ = new float64[LSTM_WEIGHTS_SIZE];
        velocity_ = new float64[LSTM_WEIGHTS_SIZE];
        memset(gradients_, 0, sizeof(float64) * LSTM_WEIGHTS_SIZE);
        memset(velocity_, 0, sizeof(float64) * LSTM_WEIGHTS_SIZE);
    }
    
    ~LSTMPredictor() {
        delete[] gradients_;
        delete[] velocity_;
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        
        initializeWeights();
        
        if (config_.useQuantization) {
            quantizeWeights();
        }
        
        bufferHead_ = 0;
        bufferSize_ = 0;
        totalPredictions_ = 0;
        predictionErrorSum_ = 0;
    }
    
    void initializeWeights() {
        for (uint32 layer = 0; layer < config_.numLayers; ++layer) {
            LSTMCell& cell = layers_[layer];
            
            float64 scale = sqrt(2.0 / (config_.inputSize + config_.hiddenSize));
            
            for (uint32 i = 0; i < config_.hiddenSize; ++i) {
                for (uint32 j = 0; j < config_.inputSize + config_.hiddenSize; ++j) {
                    cell.W_ii(i, j) = (rand() % 200 - 100) * 0.01 * scale;
                    cell.W_if(i, j) = (rand() % 200 - 100) * 0.01 * scale;
                    cell.W_ig(i, j) = (rand() % 200 - 100) * 0.01 * scale;
                    cell.W_io(i, j) = (rand() % 200 - 100) * 0.01 * scale;
                }
                
                cell.b_if[i] = 1.0;
                cell.b_ii[i] = 0;
                cell.b_ig[i] = 0;
                cell.b_io[i] = 0;
            }
        }
        
        for (uint32 i = 0; i < 6; ++i) {
            for (uint32 j = 0; j < config_.hiddenSize; ++j) {
                outputLayer_(i, j) = (rand() % 200 - 100) * 0.01;
            }
            outputBias_[i] = 0;
        }
    }
    
    void quantizeWeights() {
        for (uint32 i = 0; i < LSTM_WEIGHTS_SIZE; ++i) {
            quantizedWeights_[i] = static_cast<int16_t>(weights_[i] * 32767.0);
        }
    }
    
    void dequantizeWeights() {
        for (uint32 i = 0; i < LSTM_WEIGHTS_SIZE; ++i) {
            weights_[i] = static_cast<float64>(quantizedWeights_[i]) / 32767.0;
        }
    }
    
    void addImuData(const IMUData& imuData) {
        imuBuffer_[bufferHead_] = imuData;
        bufferHead_ = (bufferHead_ + 1) % config_.sequenceLength;
        
        if (bufferSize_ < config_.sequenceLength) {
            bufferSize_++;
        }
    }
    
    Vector3d predictAccelBias(const IMUData* sequence, uint32 length) {
        if (length < 2) return predictedAccelBias_;
        
        resetHiddenStates();
        
        for (uint32 t = 0; t < length; ++t) {
            Vector<LSTM_INPUT_SIZE, float64> input;
            input[0] = sequence[t].accel[0];
            input[1] = sequence[t].accel[1];
            input[2] = sequence[t].accel[2];
            input[3] = sequence[t].gyro[0];
            input[4] = sequence[t].gyro[1];
            input[5] = sequence[t].gyro[2];
            
            for (uint32 layer = 0; layer < config_.numLayers; ++layer) {
                layers_[layer].forward_optimized(input);
                
                for (uint32 i = 0; i < config_.hiddenSize; ++i) {
                    input[i] = layers_[layer].h[i];
                }
            }
        }
        
        Vector<6, float64> output = computeOutput();
        
        predictedAccelBias_[0] = output[0];
        predictedAccelBias_[1] = output[1];
        predictedAccelBias_[2] = output[2];
        predictedGyroBias_[0] = output[3];
        predictedGyroBias_[1] = output[4];
        predictedGyroBias_[2] = output[5];
        
        return predictedAccelBias_;
    }
    
    Vector3d predictGyroBias(const IMUData* sequence, uint32 length) {
        predictAccelBias(sequence, length);
        return predictedGyroBias_;
    }
    
    Vector<6, float64> computeOutput() {
        Vector<6, float64> output;
        
        Vector<LSTM_HIDDEN_SIZE, float64>& h = layers_[config_.numLayers - 1].h;
        
        for (uint32 i = 0; i < 6; ++i) {
            float64 sum = outputBias_[i];
            for (uint32 j = 0; j < config_.hiddenSize; ++j) {
                sum += outputLayer_(i, j) * h[j];
            }
            output[i] = sum;
        }
        
        return output;
    }
    
    void resetHiddenStates() {
        for (uint32 layer = 0; layer < config_.numLayers; ++layer) {
            layers_[layer].h = Vector<LSTM_HIDDEN_SIZE, float64>();
            layers_[layer].c = Vector<LSTM_HIDDEN_SIZE, float64>();
        }
    }
    
    void updatePredictionCovariance(const Vector3d& trueAccelBias, 
                                    const Vector3d& trueGyroBias) {
        Vector3d accelError = predictedAccelBias_ - trueAccelBias;
        Vector3d gyroError = predictedGyroBias_ - trueGyroBias;
        
        float64 error = accelError.norm() + gyroError.norm();
        lastPredictionError_ = error;
        predictionErrorSum_ += error;
        totalPredictions_++;
        
        for (uint32 i = 0; i < 3; ++i) {
            predictionCovariance_(i, i) = predictionCovariance_(i, i) * 0.99 + 
                                          accelError[i] * accelError[i] * 0.01;
            predictionCovariance_(i + 3, i + 3) = predictionCovariance_(i + 3, i + 3) * 0.99 + 
                                                   gyroError[i] * gyroError[i] * 0.01;
        }
    }
    
    void onlineLearning(const Vector3d& trueAccelBias, 
                        const Vector3d& trueGyroBias) {
        if (!config_.useOnlineLearning || bufferSize_ < config_.sequenceLength) return;
        
        Vector<6, float64> target;
        target[0] = trueAccelBias[0];
        target[1] = trueAccelBias[1];
        target[2] = trueAccelBias[2];
        target[3] = trueGyroBias[0];
        target[4] = trueGyroBias[1];
        target[5] = trueGyroBias[2];
        
        resetHiddenStates();
        
        for (uint32 t = 0; t < bufferSize_; ++t) {
            uint32 idx = (bufferHead_ + t) % config_.sequenceLength;
            
            Vector<LSTM_INPUT_SIZE, float64> input;
            input[0] = imuBuffer_[idx].accel[0];
            input[1] = imuBuffer_[idx].accel[1];
            input[2] = imuBuffer_[idx].accel[2];
            input[3] = imuBuffer_[idx].gyro[0];
            input[4] = imuBuffer_[idx].gyro[1];
            input[5] = imuBuffer_[idx].gyro[2];
            
            for (uint32 layer = 0; layer < config_.numLayers; ++layer) {
                layers_[layer].forward_optimized(input);
            }
        }
        
        Vector<6, float64> output = computeOutput();
        Vector<6, float64> error = output - target;
        
        updateOutputLayer(error);
    }
    
    void updateOutputLayer(const Vector<6, float64>& error) {
        Vector<LSTM_HIDDEN_SIZE, float64>& h = layers_[config_.numLayers - 1].h;
        
        for (uint32 i = 0; i < 6; ++i) {
            float64 grad = error[i];
            
            for (uint32 j = 0; j < config_.hiddenSize; ++j) {
                float64 weightGrad = grad * h[j] + config_.weightDecay * outputLayer_(i, j);
                outputLayer_(i, j) -= config_.learningRate * weightGrad;
            }
            
            outputBias_[i] -= config_.learningRate * grad;
        }
    }
    
    Matrix<6, 6, float64> getPredictionCovariance() const {
        return predictionCovariance_;
    }
    
    Vector3d getPredictedAccelBias() const { return predictedAccelBias_; }
    Vector3d getPredictedGyroBias() const { return predictedGyroBias_; }
    float64 getLastPredictionError() const { return lastPredictionError_; }
    float64 getAveragePredictionError() const {
        return totalPredictions_ > 0 ? predictionErrorSum_ / totalPredictions_ : 0;
    }
    
    void loadWeights(const float64* weights, uint32 size) {
        uint32 copySize = min(size, LSTM_WEIGHTS_SIZE);
        memcpy(weights_, weights, sizeof(float64) * copySize);
        
        if (config_.useQuantization) {
            quantizeWeights();
        }
    }
    
    void saveWeights(float64* weights, uint32& size) const {
        size = LSTM_WEIGHTS_SIZE;
        memcpy(weights, weights_, sizeof(float64) * size);
    }
    
    uint32 getWeightsSize() const { return LSTM_WEIGHTS_SIZE; }
    uint32 getBufferSize() const { return bufferSize_; }
    uint32 getTotalPredictions() const { return totalPredictions_; }
};

class IMUBiasPredictor {
public:
    struct Config {
        float64 windowSize;
        float64 processNoise;
        float64 measurementNoise;
        float64 initialBiasVariance;
        bool useLSTM;
        bool useAdaptiveNoise;
        
        Config() 
            : windowSize(100)
            , processNoise(1e-6)
            , measurementNoise(1e-4)
            , initialBiasVariance(1e-3)
            , useLSTM(true)
            , useAdaptiveNoise(true) {}
    };

private:
    Config config_;
    LSTMPredictor lstmPredictor_;
    
    Vector3d accelBiasEstimate_;
    Vector3d gyroBiasEstimate_;
    
    Matrix<3, 3, float64> accelBiasCovariance_;
    Matrix<3, 3, float64> gyroBiasCovariance_;
    
    Vector3d accelBuffer_[100];
    Vector3d gyroBuffer_[100];
    uint32 bufferIndex_;
    uint32 bufferCount_;
    
    float64 alpha_;
    float64 beta_;
    
    uint64 lastUpdateTime_;

public:
    IMUBiasPredictor() 
        : bufferIndex_(0)
        , bufferCount_(0)
        , alpha_(0.01)
        , beta_(0.0001)
        , lastUpdateTime_(0) {
        
        accelBiasEstimate_ = Vector3d({0, 0, 0});
        gyroBiasEstimate_ = Vector3d({0, 0, 0});
        accelBiasCovariance_ = Matrix<3, 3, float64>::identity() * 1e-3;
        gyroBiasCovariance_ = Matrix<3, 3, float64>::identity() * 1e-3;
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        
        if (config_.useLSTM) {
            LSTMPredictor::Config lstmConfig;
            lstmPredictor_.init(lstmConfig);
        }
        
        accelBiasEstimate_ = Vector3d({0, 0, 0});
        gyroBiasEstimate_ = Vector3d({0, 0, 0});
        accelBiasCovariance_ = Matrix<3, 3, float64>::identity() * config_.initialBiasVariance;
        gyroBiasCovariance_ = Matrix<3, 3, float64>::identity() * config_.initialBiasVariance;
    }
    
    void update(const IMUData& imuData) {
        accelBuffer_[bufferIndex_] = imuData.accel;
        gyroBuffer_[bufferIndex_] = imuData.gyro;
        bufferIndex_ = (bufferIndex_ + 1) % static_cast<uint32>(config_.windowSize);
        
        if (bufferCount_ < config_.windowSize) {
            bufferCount_++;
        }
        
        updateBiasEstimate(imuData);
        
        if (config_.useLSTM) {
            lstmPredictor_.addImuData(imuData);
        }
        
        lastUpdateTime_ = imuData.timestamp;
    }
    
    void updateBiasEstimate(const IMUData& imuData) {
        Matrix<3, 3, float64> Q = Matrix<3, 3, float64>::identity() * config_.processNoise;
        
        accelBiasCovariance_ = accelBiasCovariance_ + Q;
        gyroBiasCovariance_ = gyroBiasCovariance_ + Q;
        
        Vector3d accelMeasurement = imuData.accel - accelBiasEstimate_;
        Vector3d gyroMeasurement = imuData.gyro - gyroBiasEstimate_;
        
        Matrix<3, 3, float64> R = Matrix<3, 3, float64>::identity() * config_.measurementNoise;
        
        Matrix<3, 3, float64> S_accel = accelBiasCovariance_ + R;
        Matrix<3, 3, float64> S_gyro = gyroBiasCovariance_ + R;
        
        Matrix<3, 3, float64> K_accel = accelBiasCovariance_ * S_accel.inverse();
        Matrix<3, 3, float64> K_gyro = gyroBiasCovariance_ * S_gyro.inverse();
        
        accelBiasEstimate_ = accelBiasEstimate_ + K_accel * (Vector3d() - accelMeasurement) * alpha_;
        gyroBiasEstimate_ = gyroBiasEstimate_ + K_gyro * (Vector3d() - gyroMeasurement) * alpha_;
        
        Matrix<3, 3, float64> I = Matrix<3, 3, float64>::identity();
        accelBiasCovariance_ = (I - K_accel) * accelBiasCovariance_;
        gyroBiasCovariance_ = (I - K_gyro) * gyroBiasCovariance_;
    }
    
    Vector3d getAccelBias() const { return accelBiasEstimate_; }
    Vector3d getGyroBias() const { return gyroBiasEstimate_; }
    
    Vector3d getPredictedAccelBias() {
        if (config_.useLSTM && lstmPredictor_.getBufferSize() >= LSTM_SEQUENCE_LENGTH) {
            return lstmPredictor_.getPredictedAccelBias();
        }
        return accelBiasEstimate_;
    }
    
    Vector3d getPredictedGyroBias() {
        if (config_.useLSTM && lstmPredictor_.getBufferSize() >= LSTM_SEQUENCE_LENGTH) {
            return lstmPredictor_.getPredictedGyroBias();
        }
        return gyroBiasEstimate_;
    }
    
    Matrix<6, 6, float64> getPredictedBiasCovariance() {
        Matrix<6, 6, float64> cov;
        
        if (config_.useLSTM) {
            Matrix<6, 6, float64> lstmCov = lstmPredictor_.getPredictionCovariance();
            for (uint32 i = 0; i < 6; ++i) {
                for (uint32 j = 0; j < 6; ++j) {
                    cov(i, j) = lstmCov(i, j);
                }
            }
        } else {
            for (uint32 i = 0; i < 3; ++i) {
                for (uint32 j = 0; j < 3; ++j) {
                    cov(i, j) = accelBiasCovariance_(i, j);
                    cov(i + 3, j + 3) = gyroBiasCovariance_(i, j);
                }
            }
        }
        
        return cov;
    }
    
    void correctBias(const Vector3d& trueAccelBias, const Vector3d& trueGyroBias) {
        if (config_.useLSTM) {
            lstmPredictor_.onlineLearning(trueAccelBias, trueGyroBias);
            lstmPredictor_.updatePredictionCovariance(trueAccelBias, trueGyroBias);
        }
        
        accelBiasEstimate_ = trueAccelBias;
        gyroBiasEstimate_ = trueGyroBias;
    }
    
    void reset() {
        accelBiasEstimate_ = Vector3d({0, 0, 0});
        gyroBiasEstimate_ = Vector3d({0, 0, 0});
        accelBiasCovariance_ = Matrix<3, 3, float64>::identity() * config_.initialBiasVariance;
        gyroBiasCovariance_ = Matrix<3, 3, float64>::identity() * config_.initialBiasVariance;
        bufferIndex_ = 0;
        bufferCount_ = 0;
        
        if (config_.useLSTM) {
            lstmPredictor_.init();
        }
    }
    
    float64 getBiasUncertainty() const {
        float64 accelUncertainty = 0;
        float64 gyroUncertainty = 0;
        
        for (uint32 i = 0; i < 3; ++i) {
            accelUncertainty += accelBiasCovariance_(i, i);
            gyroUncertainty += gyroBiasCovariance_(i, i);
        }
        
        return sqrt(accelUncertainty + gyroUncertainty);
    }
    
    LSTMPredictor& getLSTMPredictor() { return lstmPredictor_; }
};

}

#endif
