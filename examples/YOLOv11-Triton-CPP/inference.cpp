#include "inference.hpp"

#define IOU_THRESHOLD 0.45

uint16_t float32_to_float16(float value) {
    __m128 input = _mm_set_ss(value);
    return _mm_cvtsi128_si32(_mm_cvtps_ph(input, 0));
}

float float16_to_float32(uint16_t half) {
    uint16_t sign = (half & 0x8000) >> 15;   
    uint16_t exponent = (half & 0x7C00) >> 10; 
    uint16_t mantissa = (half & 0x03FF);    

    if (exponent == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f; 
        }
        return std::ldexp(mantissa / 1024.0f, -14) * (sign ? -1.0f : 1.0f);
    } 
    else if (exponent == 31) {
        return mantissa ? NAN : (sign ? -INFINITY : INFINITY);
    }

    float real_value = std::ldexp(1.0f + mantissa / 1024.0f, exponent - 15);
    return sign ? -real_value : real_value;
}


void Image::preprocess(cv::Mat* img, std::vector<uint16_t>& triton_data, int input_w, int input_h) 
{
    int w, h, x, y;
    float r_w = input_w / (img->cols*1.0);
    float r_h = input_h / (img->rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img->rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img->cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(*img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
    unsigned char* data = (unsigned char*)out.data;
    int step = out.step;

    for (int y = 0; y < input_h; ++y)
    {
        for (int k = 0; k < 3; ++k)
        {   
            for (int x = 0; x < input_w; ++x)
            {
                float temp_f = data[y * step + x * 3 + k] / 255.0f;

                triton_data[k * input_w * input_h + y * input_w + x] = float32_to_float16(temp_f);

            }
        }
    }
}


int getDetectionsFromTritonRawData(std::vector<float>& detection_results, std::vector<struct detection_struct> &tespitler, std::vector<std::string>& object_class_list, float confidence_threshold, int IMAGE_HEIGHT, int IMAGE_HEIGHT)
{
    const size_t shape[3] = {1, object_class_list.size()+4, 8400};
	std::vector<BoundingBox> boxes;
    for (size_t i = 0; i < shape[2]; i++) 
    { 

		int x = int(detection_results[0 * shape[2] + i]);
		int y = int(detection_results[1 * shape[2] + i]);
		int w = int(detection_results[2 * shape[2] + i]);
		int h = int(detection_results[3 * shape[2] + i]);
		
		for(size_t j =0 ; j < object_class_list.size(); j++)
		{
			if (detection_results[(4+j) * shape[2] + i] > 0.01)
			{
				BoundingBox box;
				box.x = static_cast<float>(x);
				box.y = static_cast<float>(y);
				box.w = static_cast<float>(w);
				box.h = static_cast<float>(h);
				box.score = detection_results[(4 + j) * shape[2] + i];
				box.class_id = j;  // j, 0 ile num_classes-1 arasında bir değer alır
				boxes.push_back(box);
			}

		}
   
    }
	auto nms_boxes = NMS(boxes, IOU_THRESHOLD);
	tespitler.clear();
	if(nms_boxes.size()==0)
	{
		return 0;
	}
	float scale_x = 0.0;
	float scale_y = 0.0;

	int x1,y1 = 0;
	int x2,y2 = 0;

	float shift_factor_x = 0.6;
    float shift_factor_y = 0.5;  

	int offset_shift = (IMAGE_HEIGHT/640.0f)*10;

	if (IMAGE_HEIGHT<=640)
	{
		scale_x = static_cast<float>(IMAGE_HEIGHT - 640.0f ) * 0.5 ;  
		scale_y = static_cast<float>(IMAGE_HEIGHT - 640.0f) * 0.5 ;
	}
	for (size_t i = 0; i < nms_boxes.size(); ++i)
	{
		if (nms_boxes[i].score< confidence_threshold)
		{
			continue;
		}
		struct detection_struct tespit_yapi ;
		tespitler.push_back(tespit_yapi);
		tespitler[tespitler.size() - 1].confidence_score = nms_boxes[i].score;
		if (IMAGE_HEIGHT==640)
		{
			scale_x = static_cast<float>(IMAGE_HEIGHT - 640.0f ) * 0.5 ;  
			scale_y = static_cast<float>(IMAGE_HEIGHT - 640.0f) * 0.5 ;
			x1 = static_cast<int>((nms_boxes[i].x - nms_boxes[i].w/2) + scale_x);
			y1 = static_cast<int>((nms_boxes[i].y - nms_boxes[i].h/2) + scale_y) ;
			x2 = static_cast<int>((nms_boxes[i].x + nms_boxes[i].w/2) + scale_x);
			y2 = static_cast<int>((nms_boxes[i].y + nms_boxes[i].h/2) + scale_y);
		}
		else if(IMAGE_HEIGHT>=1080)
		{
			x1 = static_cast<int>((nms_boxes[i].x - nms_boxes[i].w/2) * (IMAGE_HEIGHT/640) );
			y1 = static_cast<int>((nms_boxes[i].y - nms_boxes[i].h/2) * (IMAGE_HEIGHT/640) - ((IMAGE_HEIGHT - IMAGE_HEIGHT) / 2.0)) ;
			x2 = static_cast<int>((nms_boxes[i].x + nms_boxes[i].w/2) * (IMAGE_HEIGHT/640) );
			y2 = static_cast<int>((nms_boxes[i].y + nms_boxes[i].h/2) * (IMAGE_HEIGHT/640)- ((IMAGE_HEIGHT - IMAGE_HEIGHT) / 2.0));
		}

		float x_center, y_center, width, height;
		x_center = (x1 + x2) / 2.0f;
		y_center = (y1 + y2) / 2.0f;
		width = x2 - x1;
		height = y2 - y1;
		tespitler[tespitler.size() - 1].bbox.x = x_center - width/2 ;
		tespitler[tespitler.size() - 1].bbox.y = y_center - height/2;
		tespitler[tespitler.size() - 1].bbox.width  = width ;
		tespitler[tespitler.size() - 1].bbox.height =  height;  
		if (tespitler[tespitler.size() - 1].bbox.x <= 0)
			tespitler[tespitler.size() - 1].bbox.x = offset_shift;
		if (tespitler[tespitler.size() - 1].bbox.y <= 0)
			tespitler[tespitler.size() - 1].bbox.y = offset_shift;
		if (tespitler[tespitler.size() - 1].bbox.x + tespitler[tespitler.size() - 1].bbox.width  >= IMAGE_HEIGHT)
		{
			tespitler[tespitler.size() - 1].bbox.width  -= tespitler[tespitler.size() - 1].bbox.x + tespitler[tespitler.size() - 1].bbox.width  - IMAGE_HEIGHT + offset_shift ;
		}
		if (tespitler[tespitler.size() - 1].bbox.y + tespitler[tespitler.size() - 1].bbox.height >= IMAGE_HEIGHT)
		{	
			tespitler[tespitler.size() - 1].bbox.height -= tespitler[tespitler.size() - 1].bbox.y + tespitler[tespitler.size() - 1].bbox.height - IMAGE_HEIGHT + offset_shift ;
		}
		// NesneIsmi
		tespitler[tespitler.size() - 1].name = object_class_list[nms_boxes[i].class_id];
		// No
		tespitler[tespitler.size() - 1].class_id = nms_boxes[i].class_id;
	}
	return 0;
}

float IoU(const BoundingBox& box1, const BoundingBox& box2) {
    float x1_min = box1.x - box1.w / 2.0f;
    float y1_min = box1.y - box1.h / 2.0f;
    float x1_max = box1.x + box1.w / 2.0f;
    float y1_max = box1.y + box1.h / 2.0f;

    float x2_min = box2.x - box2.w / 2.0f;
    float y2_min = box2.y - box2.h / 2.0f;
    float x2_max = box2.x + box2.w / 2.0f;
    float y2_max = box2.y + box2.h / 2.0f;

    float inter_x_min = std::max(x1_min, x2_min);
    float inter_y_min = std::max(y1_min, y2_min);
    float inter_x_max = std::min(x1_max, x2_max);
    float inter_y_max = std::min(y1_max, y2_max);

    float inter_width = inter_x_max - inter_x_min;
    float inter_height = inter_y_max - inter_y_min;

    if (inter_width <= 0 || inter_height <= 0)
        return 0.0f;

    float inter_area = inter_width * inter_height;
    float area1 = (x1_max - x1_min) * (y1_max - y1_min);
    float area2 = (x2_max - x2_min) * (y2_max - y2_min);
    float union_area = area1 + area2 - inter_area;

    return inter_area / union_area;
}

std::vector<BoundingBox> NMS(const std::vector<BoundingBox>& boxes, float iou_threshold) {

    std::vector<BoundingBox> result;

    std::vector<BoundingBox> sorted_boxes = boxes;
    std::sort(sorted_boxes.begin(), sorted_boxes.end(), [](const BoundingBox& a, const BoundingBox& b) {
        return a.score > b.score;
    });
    
    std::vector<bool> suppressed(sorted_boxes.size(), false);

    for (size_t i = 0; i < sorted_boxes.size(); ++i) {
        if (suppressed[i])
            continue;

        result.push_back(sorted_boxes[i]);

        for (size_t j = i + 1; j < sorted_boxes.size(); ++j) {
            if (suppressed[j])
                continue;
            if (IoU(sorted_boxes[i], sorted_boxes[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

TritonCommunication::TritonCommunication(std::string triton_address, std::string model_name, std::string model_version, int image_channel, int image_width, int image_height,int class_count) : options(model_name)
{

    triton::client::Error err;
    this->triton_url = triton_address;
    
    this->options.model_version_ = model_version;

    this->shape = {1, image_channel, image_width, image_height}; 
    this->input_byte_size = image_channel * image_width * image_height * sizeof(uint16_t) ;
    this->output_byte_size = 1 * 6 *8400 * sizeof(uint16_t);
    err = tc::InferenceServerGrpcClient::Create(&(this->client), this->triton_url);

    err = tc::InferenceServerGrpcClient::Create(&(this->client), this->triton_url);
    if (!err.IsOk()) {
        std::cout<< "Create grpc client error:"<<err.Message()<<std::endl;
    }
    bool live;
    err = client->IsServerReady(&live);
    if (!err.IsOk() || !live) {
        std::cout<< "Triton server is not live !"<<std::endl;
        exit(-1);
    }

    bool model_ready;
    err = client->IsModelReady(&model_ready,model_name,model_version);
    if (!err.IsOk() || !model_ready) {
        std::cerr << "Model:[" << model_name << "] has not been deployed on Triton Server. Triton Server Adress:["<<triton_address <<"]"<<std::endl;
        exit(-1);
    }

    std::cout<<"Triton server is LIVE and model is READY!"<<std::endl;
}

void TritonCommunication::infer(uint16_t* image_data)
{
    size_t num_elements = output_byte_size / sizeof(uint16_t);
    tc::Error err;         
    tc::InferInput* input0;
    
    err = tc::InferInput::Create(&input0, "images", shape, "FP16");
    std::shared_ptr<tc::InferInput> input0_ptr;
    input0_ptr.reset(input0);

    err = input0_ptr->AppendRaw((const uint8_t*)image_data, this->input_byte_size);

    std::vector<tc::InferInput*> inputs = {input0_ptr.get()};

    tc::InferResult* results;
    err = client->Infer(&results, options, inputs);
    results_ptr.reset(results);

    float *output0_data;
    size_t output0_byte_size;
    std::vector<uint16_t> result_fp16_raw_data;

    results->RawData("output0", (const uint8_t**)&output0_data, &output0_byte_size);
    result_fp16_raw_data.resize(output0_byte_size/sizeof(uint16_t));

    std::memcpy(result_fp16_raw_data.data(), output0_data, output0_byte_size);
    std::vector<float> float32_data(num_elements);

    for (size_t i = 0; i < num_elements; i++) {
        float32_data[i] = float16_to_float32(result_fp16_raw_data[i]);   
    }

    output_raw_data = float32_data;

}