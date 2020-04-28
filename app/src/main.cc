#include <iostream>
#include <experimental/filesystem>
#include <vector>
#include <queue>
#include <map>
#include <chrono>
#include <unistd.h>
#include <stdlib.h>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <opencv2/opencv.hpp>
#include <dnndk/dnndk.h>

using namespace std;
using namespace cv;

#define KRENEL "centernet"
#define INPUT_NODE "extractor_model_conv1_Conv2D"
#define OUTPUT_NODE "detector_model_conv2d_12_Conv2D"
#define NUM_CLASSES 6
#define NUM_IMAGES 6355

#define IMG_HEIGHT 1216
#define IMG_WIDTH 1936
#define SCALE_RATIO 4
#define STRIDE 4
#define VALID_HEIGHT (IMG_HEIGHT / SCALE_RATIO)
#define VALID_WIDTH (IMG_WIDTH / SCALE_RATIO)
#define DEBUG false
#define TRAIN_COMMAND false
#define DATA_PATH "dtc_test_images/"
#include "conv_last.cc"

struct Detection
{
    string stem;
    int left;
    int top;
    int right;
    int bottom;
    string categ;
    float logit_score;
};

typedef priority_queue<Detection *, vector<Detection *>, function<bool(Detection *, Detection *)>> DetQueue;

int8_t normalize_and_quantize(int pixel, int i, int j, int k, float scale)
{
    int8_t q_pixel;

    if ((i < VALID_HEIGHT) && (j < VALID_WIDTH))
        q_pixel = pixel / 255.0 * scale;
    else
        q_pixel = 0;
    return q_pixel;
}

void setInputImage(DPUTask *task, const char *inNode, const cv::Mat &image)
{
    DPUTensor *in = dpuGetInputTensor(task, inNode);
    float scale = dpuGetTensorScale(in);
    int w = dpuGetTensorWidth(in);
    int h = dpuGetTensorHeight(in);
    int c = 3;
    int8_t *data = dpuGetTensorAddress(in);
    image.forEach<Vec3b>([&](Vec3b &p, const int pos[2]) -> void {
        int start = pos[0] * w * c + pos[1] * c;
        for (int k = 0; k < 3; k++)
            data[start + k] = normalize_and_quantize(p[0], pos[0], pos[1], k, scale);
    });
}

Mat read_image(string stem)
{
    string imageFile = DATA_PATH + stem + ".jpg";
    Mat img = imread(imageFile);
    assert(img.data != NULL);
    return img;
}

void preprocessing(Mat &img)
{
    cvtColor(img, img, COLOR_BGR2RGB);
    resize(img, img, Size(VALID_WIDTH, VALID_HEIGHT));
}

void postprocessing(float *output_conv, float *output, DetQueue &dets, string stem, int h, int w, int c)
{
    conv_last(output_conv, output);
    map<int, string> id2categ = {
        {0, "Car"},
        {1, "Truck"},
        {2, "Bicycle"},
        {3, "Pedestrian"},
        {4, "Signal"},
        {5, "Signs"},
    };

    int p, q;
    bool **peak = new bool *[h];
    for (int i = 0; i < h; i++)
        peak[i] = new bool[w]{false};

    int H = VALID_HEIGHT / STRIDE;
    int W = VALID_WIDTH / STRIDE;
    for (int k = 4; k < c; k++)
    {
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++)
                peak[i][j] = true;

        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++)
            {
                p = i * w * c + j * c;

                if (!peak[i][j])
                    continue;

                for (int y = -1; y <= 1; y++)
                    for (int x = -1; x <= 1; x++)
                    {
                        if (x == 0 && y == 0)
                            continue;

                        if (i + y < 0 || i + y >= H || j + x < 0 || j + x >= W)
                            continue;

                        q = (i + y) * w * c + (j + x) * c;

                        auto diff = output_conv[p + k] - output_conv[q + k];
                        if (diff < 0)
                            peak[i][j] = false;
                        else if (diff > 0)
                            peak[i + y][j + x] = false;
                    }

                if (!peak[i][j])
                    continue;

                float box_width = exp(output_conv[p]);
                float box_height = exp(output_conv[p + 1]);
                float offset_x = tanh(output_conv[p + 2]);
                float offset_y = tanh(output_conv[p + 3]);
                float logit_score = output_conv[p + k];

                auto det = new Detection({
                    stem,
                    (int)(std::max((float)0, j + offset_x - box_width / 2) * STRIDE * SCALE_RATIO),
                    (int)(std::max((float)0, i + offset_y - box_height / 2) * STRIDE * SCALE_RATIO),
                    (int)(std::min((float)w, j + offset_x + box_width / 2) * STRIDE * SCALE_RATIO),
                    (int)(std::min((float)h, i + offset_y + box_height / 2) * STRIDE * SCALE_RATIO),
                    id2categ[k - 4],
                    logit_score,
                });

                dets.push(det);
            }

        if (DEBUG)
        {
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                    cout << (int)peak[i][j];
                cout << endl;
            }
            cout << endl;
        }
    }

    for (int i = 0; i < h; i++)
        delete[] peak[i];
    delete[] peak;
}

void print_elapsed_time(chrono::system_clock::time_point start, string title)
{
    auto diff = chrono::system_clock::now() - start;
    cout << title << " :\t" << chrono::duration_cast<chrono::milliseconds>(diff).count() << "[msec]" << endl;
}

void run(DPUTask *task, string output_file_name, vector<string> &stems)
{
    std::mutex start_mtx;
    std::condition_variable start_cond;
    std::mutex finish_mtx;
    std::condition_variable finish_cond;
    bool finish = false;
    bool enqueue = false;
    bool done = false;
    int d_dpu_time = 0;
    std::thread dpu_run_thread([&] {
        while (true)
        {
            {
                std::unique_lock<std::mutex> lk(start_mtx);
                start_cond.wait(lk);
                if (finish)
                    return;
                if (!enqueue)
                    continue;
                enqueue = false;
            }
            dpuRunTask(task);
            d_dpu_time = dpuGetTaskProfile(task) / 1000;
            {
                lock_guard<mutex> lk(finish_mtx);
                done = true;
            }
            finish_cond.notify_one();
        }
    });
    auto start_dpu = [&]() {
        {
            lock_guard<mutex> lk(start_mtx);
            enqueue = true;
        }
        start_cond.notify_one();
    };
    auto wait_dpu = [&]() {
        std::unique_lock<std::mutex> lk(finish_mtx);
        finish_cond.wait(lk, [&] { return done; });
        done = false;
        return d_dpu_time;
    };
    unsigned N = stems.size();
    assert(task);
    cout << "Get tensor info" << endl;
    DPUTensor *outputTensor = dpuGetOutputTensor(task, OUTPUT_NODE);
    int w = dpuGetTensorWidth(outputTensor);
    int h = dpuGetTensorHeight(outputTensor);
    int c = 10;
    cout << "h : " << h << ", w : " << w << ", c : " << c << endl;

    ofstream meta("metadata.csv");
    meta << "h,w,c" << endl;
    meta << h << "," << w << "," << c << endl;
    meta.close();

    cerr << "loading image" << endl;
    ofstream data(output_file_name);
    data << "stem,left,top,right,bottom,categ,logit_score" << endl;

    vector<Mat> images;
    int out_size = h * w * 64;
    float32_t *output = (float32_t *)aligned_alloc(64, sizeof(float32_t) * 80 * 128 * 64);
    float32_t *output_conv = (float32_t *)aligned_alloc(64, sizeof(float32_t) * 80 * 128 * 10);
    images.reserve(50);
    for (auto stem : stems)
        images.push_back(read_image(stem));

    unsigned pre_time = 0;
    unsigned dpu_time = 0;
    unsigned post_time = 0;

    cerr << "run" << endl;
    auto start = chrono::system_clock::now();
    {
        auto s = chrono::system_clock::now();
        preprocessing(images[0]);
        setInputImage(task, INPUT_NODE, images[0]);
        pre_time += chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - s).count();
    }

    start_dpu();
    for (int i = 1; i < N; ++i)
    {
        cerr << stems[i - 1] << endl;
        {
            auto s = chrono::system_clock::now();
            preprocessing(images[i]);
            pre_time += chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - s).count();
        }
        dpu_time += wait_dpu();
        dpuGetOutputTensorInHWCFP32(task, OUTPUT_NODE, output, out_size);
        setInputImage(task, INPUT_NODE, images[i]);
        start_dpu();
        DetQueue dets([](Detection *a, Detection *b) { return a->logit_score < b->logit_score; });
        {
            auto s = chrono::system_clock::now();
            postprocessing(output_conv, output, dets, stems[i - 1], h, w, c);
            post_time += chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - s).count();
        }
        Detection *det;
        map<string, int> counter;
        while (dets.size() > 0)
        {
            det = dets.top();
            if (counter[det->categ] < 100)
            {
                counter[det->categ]++;

                data << det->stem << ","
                     << det->left << ","
                     << det->top << ","
                     << det->right << ","
                     << det->bottom << ","
                     << det->categ << ","
                     << det->logit_score << endl;
            }
            dets.pop();
            delete det;
        }
    }

    cerr << stems[N - 1] << endl;
    cerr << "waiting.." << endl;
    dpu_time += wait_dpu();
    dpuGetOutputTensorInHWCFP32(task, OUTPUT_NODE, output, out_size);
    DetQueue dets([](Detection *a, Detection *b) { return a->logit_score < b->logit_score; });
    {
        auto s = chrono::system_clock::now();
        postprocessing(output_conv, output, dets, stems[N - 1], h, w, c);
        post_time += chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - s).count();
    }
    Detection *det;
    map<string, int> counter;
    while (dets.size() > 0)
    {
        det = dets.top();
        if (counter[det->categ] < 100)
        {
            counter[det->categ]++;

            data << det->stem << ","
                 << det->left << ","
                 << det->top << ","
                 << det->right << ","
                 << det->bottom << ","
                 << det->categ << ","
                 << det->logit_score << endl;
        }
        dets.pop();
        delete det;
    }
    cerr << "done" << endl;
    auto total_time = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count();
    std::cout << "average_total:" << (int)(total_time / N) << "[ms]" << endl;
    std::cout << "average_pre:" << (int)(pre_time / N) << "[ms]" << endl;
    std::cout << "average_dpu:" << (int)(dpu_time / N) << "[ms]" << endl;
    std::cout << "average_post:" << (int)(post_time / N) << "[ms]" << endl;

    data.close();
    free(output);
    free(output_conv);
    {
        std::lock_guard<std::mutex> lk(start_mtx);
        finish = true;
        start_cond.notify_one();
    }
    dpu_run_thread.join();
}

int main(int argc, char *argv[])
{
    opterr = 0;
#if DEBUG
    cout << "DEBUG MODE" << endl;
#endif
    vector<string> stems;
    stems.reserve(50);

    string output_file = string(argv[1]);
#ifdef TRAIN_COMMAND
    for (int i = 2; i < argc; ++i)
    {
        stems.push_back(argv[i]);
    }
    if (stems.size() > 50)
    {
        cout << "too many images. (max:50 images)" << endl;
        return -1;
    }
#else
    int start_no = atoi(argv[2]);
    int end_no = atoi(argv[3]);
    if (end_no - start_no > 50)
    {
        cout << "too many images. (max:50 images)" << endl;
        return -1;
    }
    for (int i = start_no; i < end_no; ++i)
    {
        stringstream ss;
        ss << "test_" << setfill('0') << setw(4) << i;
        stems.push_back(ss.str());
    }
#endif
    DPUKernel *kernel;
    DPUTask *task;
    dpuOpen();
    kernel = dpuLoadKernel(KRENEL);
    task = dpuCreateTask(kernel, 0);
    run(task, output_file, stems);
    dpuDestroyTask(task);
    dpuDestroyKernel(kernel);
    dpuClose();
    return 0;
}
