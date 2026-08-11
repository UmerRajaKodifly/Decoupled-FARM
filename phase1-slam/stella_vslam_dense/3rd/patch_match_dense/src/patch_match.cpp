#include "patch_match.hpp"
#include <opencv2/imgproc.hpp>
#include <chrono>

using namespace std::literals::chrono_literals;

namespace dense
{

PatchMatch::PatchMatch(std::shared_ptr<Config> config)
{
	this->estimator = std::make_unique<DepthmapEstimator>();
	this->cleaner = std::make_unique<DepthmapCleaner>();
	this->pruner = std::make_unique<DepthmapPruner>();

	if (config)
	{
		this->estimator->SetMinPatchSD(config->min_patch_standard_deviation);
		this->estimator->SetPatchSize(config->patch_size);
		this->estimator->SetPatchMatchIterations(config->patchmatch_iterations);
		this->estimator->SetMinScore(config->min_score);
		this->cleaner->SetMinConsistentViews(config->min_consistent_views);
		this->cleaner->SetDepthQueueSize(config->depthmap_queue_size);
		this->cleaner->SetSameDepthThreshold(config->depthmap_same_depth_threshold);
		this->pruner->SetMinViews(config->min_views);
		this->pruner->SetDepthQueueSize(config->pointcloud_queue_size);
		this->pruner->SetSameDepthThreshold(config->pointcloud_same_depth_threshold);

		this->num_init = -(config->depthmap_queue_size + config->pointcloud_queue_size);
		this->num_drop = config->depthmap_queue_size / 2 + 1;
		this->min_stereo_score = config->min_stereo_score;
	}

	this->count = this->num_init;

	this->v_img1.resize(2);
	this->v_img2.resize(this->num_drop);

	this->i_img1 = this->v_img1.begin();
	this->i_img2 = this->v_img2.begin();

	this->t_estimator = std::thread(std::bind(&PatchMatch::run_estimator, this));
	this->t_cleaner = std::thread(std::bind(&PatchMatch::run_cleaner, this));
	this->t_pruner = std::thread(std::bind(&PatchMatch::run_pruner, this));
}

PatchMatch::~PatchMatch()
{
	this->alive = false;
	this->t_estimator.join();
	this->t_cleaner.join();
	this->t_pruner.join();
}

PatchMatch::ReturnValue PatchMatch::addView(const cv::Matx33d &R, const cv::Vec3d &t, const cv::Mat &img,
		const cv::Mat &mask, const std::vector<cv::Vec3d> &sparse)
{
	static std::shared_ptr<DepthmapInput> input;
	if (input && this->score_stereo(input->t, input->sparse, t, sparse) < this->min_stereo_score)
		return NOT_ENOUGH_STEREO;
	input = std::make_shared<DepthmapInput>();
	input->R = R;
	input->t = t;
	if (img.type() == CV_8UC3)
	{
		cv::cvtColor(img, input->grey, cv::COLOR_BGR2GRAY);
		input->color = img;
	}
	else
	{
		input->grey = img;
		cv::cvtColor(img, input->color, cv::COLOR_GRAY2BGR);
	}
	input->mask = mask;
	input->sparse = sparse;
	{
		std::lock_guard<std::mutex> lock(this->m_input);
		this->q_input.push(input);
	}
	this->count++;
	return SUCCESS;
}

PatchMatch::ReturnValue PatchMatch::getDepth(cv::Mat &depthmap,
	std::vector<cv::Vec3d> &dense, std::vector<cv::Vec3b> &color, bool blocking)
{
	if (this->count < 0) return NOT_ENOUGH_IMAGES;
	if (this->count == 0)
	{
		if (this->initializing)
		{
			return NOT_ENOUGH_IMAGES;
		}
		else return NO_OPEN_REQUEST;
	}
	this->initializing = false;
	bool ready;
	DepthmapPrunerResult *result;
	{
		std::lock_guard<std::mutex> lock(this->m_output);
		ready = !this->q_output.empty();
		if (ready)
		{
			result = this->q_output.front();
			this->q_output.pop();
		}
	}
	if (!ready && blocking)
	{
		while (!ready)
		{
			{
				std::lock_guard<std::mutex> lock(this->m_output);
				ready = !this->q_output.empty();
				if (ready)
				{
					result = this->q_output.front();
					this->q_output.pop();
				}
			}
			std::this_thread::yield();
		}
	}
	if (ready)
	{
		depthmap = result->pruned_depth;
		dense = result->points3d;
		color = result->colors;
		this->count--;
		return SUCCESS;
	}
	else
		return NOT_READY;
	return UNDEFINED;
}

PatchMatch::ReturnValue PatchMatch::calculateDepth(const cv::Matx33d &R, const cv::Vec3d &t,
		const cv::Mat &img, const cv::Mat &mask, const std::vector<cv::Vec3d> &sparse,
		cv::Mat &depthmap, std::vector<cv::Vec3d> &dense, std::vector<cv::Vec3b> &color)
{
	ReturnValue retval = this->addView(R, t, img, mask, sparse);
	if (retval != SUCCESS)
		return retval;
	else
		return this->getDepth(depthmap, dense, color, true);
}

bool PatchMatch::isIdle()
{
	return this->count <= 0;
}

uint8_t PatchMatch::numPrerun()
{
	return -this->num_init;
}

uint8_t PatchMatch::numDropped()
{
	return this->num_drop;
}

void PatchMatch::reset()
{
	std::lock_guard<std::mutex> lock_estimator(this->m_estimator);
	std::lock_guard<std::mutex> lock_cleaner(this->m_cleaner);
	std::lock_guard<std::mutex> lock_pruner(this->m_pruner);
	std::lock_guard<std::mutex> lock_input(this->m_input);
	std::lock_guard<std::mutex> lock_hidden1(this->m_hidden1);
	std::lock_guard<std::mutex> lock_hidden2(this->m_hidden2);
	std::lock_guard<std::mutex> lock_output(this->m_output);
	this->q_input = std::queue<std::shared_ptr<DepthmapInput>>();
	this->q_hidden1 = std::queue<DepthmapEstimatorResult*>();
	this->q_hidden2 = std::queue<DepthmapCleanerResult*>();
	this->q_output = std::queue<DepthmapPrunerResult*>();
	this->estimator->reset();
	this->cleaner->reset();
	this->pruner->reset();
	this->count = this->num_init;
}

void PatchMatch::run_estimator()
{
	while (this->alive)
	{
		std::this_thread::yield();
		std::lock_guard<std::mutex> lock_estimator(this->m_estimator);
		bool has_new_input;
		std::shared_ptr<DepthmapInput> input;
		{
			std::lock_guard<std::mutex> lock(this->m_input);
			has_new_input = !this->q_input.empty();
			if (has_new_input)
			{
				input = this->q_input.front();
				this->q_input.pop();
			}
		}
		if (has_new_input)
		{
			double max_depth = this->compute_max_depth_from_sparse(input->R, input->t, input->sparse);
			this->estimator->AddView(input->R, input->t, input->grey, input->mask, input->sparse, max_depth, this->min_depth);
			DepthmapEstimatorResult *result = new DepthmapEstimatorResult();
			*this->i_img1 = input->color;
			++this->i_img1;
			if (this->i_img1 == this->v_img1.end())
				this->i_img1 = this->v_img1.begin();
			result->color = *this->i_img1;
			if (this->estimator->IsReadyForCompute())
			{
				this->estimator->ComputePatchMatch(result);
				{
					std::lock_guard<std::mutex> lock(this->m_hidden1);
					this->q_hidden1.push(result);
				}
				this->estimator->PopHeadUnlessOne();
			}
		}
	}
}

void PatchMatch::run_cleaner()
{
	while (this->alive)
	{
		std::this_thread::yield();
		std::lock_guard<std::mutex> lock_cleaner(this->m_cleaner);
		bool has_new_input;
		DepthmapEstimatorResult *input;
		{
			std::lock_guard<std::mutex> lock(this->m_hidden1);
			has_new_input = !this->q_hidden1.empty();
			if (has_new_input)
			{
				input = this->q_hidden1.front();
				this->q_hidden1.pop();
			}
		}
		if (has_new_input)
		{
			this->cleaner->AddView(input);
			DepthmapCleanerResult *result = new DepthmapCleanerResult();
			*this->i_img2 = input->color;
			++this->i_img2;
			if (this->i_img2 == this->v_img2.end())
				this->i_img2 = this->v_img2.begin();
			result->color = *this->i_img2;
			delete input;
			if (this->cleaner->IsReadyForCompute())
			{
				this->cleaner->Clean(result);
				{
					std::lock_guard<std::mutex> lock(this->m_hidden2);
					this->q_hidden2.push(result);
				}
				this->cleaner->PopHeadUnlessOne();
			}
		}
	}
}

void PatchMatch::run_pruner()
{
	while (this->alive)
	{
		std::this_thread::yield();
		std::lock_guard<std::mutex> lock_pruner(this->m_pruner);
		bool has_new_input;
		DepthmapCleanerResult *input;
		{
			std::lock_guard<std::mutex> lock(this->m_hidden2);
			has_new_input = !this->q_hidden2.empty();
			if (has_new_input)
			{
				input = this->q_hidden2.front();
				this->q_hidden2.pop();
			}
		}
		if (has_new_input)
		{
			this->pruner->AddView(input, input->color);
			delete input;
			DepthmapPrunerResult *result = new DepthmapPrunerResult();
			if (this->pruner->IsReadyForCompute())
			{
				this->pruner->Prune(result);
				{
					std::lock_guard<std::mutex> lock(this->m_output);
					this->q_output.push(result);
				}
				this->pruner->PopHead();
			}
		}
	}
}

double PatchMatch::compute_max_depth_from_sparse(const cv::Matx33d &R, const cv::Vec3d &t,
	const std::vector<cv::Vec3d> &sparse)
{
	double perc_90;
	std::vector<float> depths;
	depths.reserve(sparse.size());
	for (const cv::Vec3d &point : sparse)
	{
		depths.push_back(cv::norm(R * point + t));
	}
	std::sort(depths.begin(), depths.end());
	perc_90 = depths[depths.size() / 10 * 9];

	return perc_90 * 1.1;
}

bool PatchMatch::compare_points(const cv::Vec3d *p1, const cv::Vec3d *p2)
{
	return p1->dot(*p1) < p2->dot(*p2);
}

double PatchMatch::score_stereo(const cv::Vec3d &t1, const std::vector<cv::Vec3d> &sparse1,
		const cv::Vec3d &t2, const std::vector<cv::Vec3d> &sparse2)
{
	double theta_min = M_PI / 60;
	double theta_max = M_PI / 6;

	std::vector<const cv::Vec3d*> sparse1p, sparse2p;
	sparse1p.reserve(sparse1.size());
	sparse2p.reserve(sparse2.size());
	for (const cv::Vec3d &point : sparse1)
		sparse1p.push_back(&point);
	for (const cv::Vec3d &point : sparse2)
		sparse2p.push_back(&point);
	std::sort(sparse1p.begin(), sparse1p.end(), compare_points);
	std::sort(sparse2p.begin(), sparse2p.end(), compare_points);

	std::vector<const cv::Vec3d*> sparse_common;
	std::set_intersection(sparse1p.begin(), sparse1p.end(), sparse2p.begin(), sparse2p.end(), std::back_inserter(sparse_common), compare_points);
	if (sparse_common.empty())
		return 0.;

	double score = 0.;
	for (const cv::Vec3d *point : sparse_common)
	{
		cv::Vec3d dist1 = t1 - *point;
		cv::Vec3d dist2 = t2 - *point;
		double nom = dist2.dot(dist1);
		double var1 = dist1.dot(dist1);
		double var2 = dist2.dot(dist2);

		double theta = std::acos(nom / std::sqrt(var1 * var2));
		if (theta > theta_min && theta < theta_max)
		{
			score += 1.;
		}
	}
	return score / sparse_common.size();
}

const char *PatchMatch::retval_to_cstr(ReturnValue retval)
{
	switch (retval)
	{
	case SUCCESS:
		return "Success";
	case NO_OPEN_REQUEST:
		return "No open request";
	case NOT_ENOUGH_IMAGES:
		return "Not enough images to calculate a output";
	case NOT_READY:
		return "Computation has not finished";
	case NOT_ENOUGH_STEREO:
		return "Not enough stereo between views";
	case UNDEFINED:
		return "Undefined";
	default:
		return "Unknown return value";
	}
}

}
