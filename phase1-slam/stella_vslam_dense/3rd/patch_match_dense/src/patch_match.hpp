#pragma once

#include "depthmap.hpp"
#include <queue>
#include <vector>
#include <thread>
#include <opencv2/core.hpp>

namespace dense
{

struct DepthmapInput
{
	cv::Matx33d R;
	cv::Vec3d t;
	cv::Mat color;
	cv::Mat grey;
	cv::Mat mask;
	std::vector<cv::Vec3d> sparse;
};

class PatchMatch
{
public:
	enum ReturnValue
	{
		SUCCESS,
		NO_OPEN_REQUEST,
		NOT_ENOUGH_IMAGES,
		NOT_READY,
		NOT_ENOUGH_STEREO,
		UNDEFINED
	};
	/**
	 * @brief Converts ReturnValues to C strings
	 *
	 * @param retval ReturnValue
	 * @return const char*
	 */
	static const char *retval_to_cstr(ReturnValue retval);

	struct Config
	{
		// estimator
		float min_patch_standard_deviation;
		int patch_size; // > 3 && % 2 == 1
		int patchmatch_iterations;
		double min_score;
		// cleaner
		int min_consistent_views;
		size_t depthmap_queue_size;
		float depthmap_same_depth_threshold;
		// pruner
		int min_views;
		size_t pointcloud_queue_size;
		float pointcloud_same_depth_threshold;
		// this
		double min_stereo_score; // >= 0 && <= 1
	};

	/**
	 * @brief Construct a new Patch Match Dense object
	 *
	 * @param config (optional) configuration
	 */
	PatchMatch(std::shared_ptr<Config> config = nullptr);
	~PatchMatch();

	/**
	 * @brief Add a view for depth calculation
	 *
	 * @param R rotation matix from view to global
	 * @param t translation vector from view to global
	 * @param img image of view
	 * @param mask image mask
	 * @param sparse sparse point cloud of view in global
	 * @return ReturnValue status information
	 */
	ReturnValue addView(const cv::Matx33d &R, const cv::Vec3d &t, const cv::Mat &img,
		const cv::Mat &mask, const std::vector<cv::Vec3d> &sparse);

	/**
	 * @brief Try to get the next depthmap and point cloud blocking or non blocking
	 *
	 * @param depthmap next available depthmap
	 * @param dense point cloud to depthap
	 * @param color corresponding color for point cloud
	 * @param blocking whether to wait for the output or not
	 * @return ReturnValue status information
	 */
	ReturnValue getDepth(cv::Mat &depthmap, std::vector<cv::Vec3d> &dense,
		std::vector<cv::Vec3b> &color, bool blocking = true);

	/**
	 * @brief Add a view and get next possible depthmap and point cloud
	 *
	 * Simply calls addView and getDepth if possible.
	 *
	 * @param R rotation matix from view to global
	 * @param t translation vector from view to global
	 * @param img image of view
	 * @param mask image mask
	 * @param sparse sparse point cloud of view in global
	 * @param depthmap next available depthmap
	 * @param dense point cloud to depthap
	 * @param color corresponding color for point cloud
	 * @return ReturnValue status information
	 */
	ReturnValue calculateDepth(const cv::Matx33d &R, const cv::Vec3d &t, const cv::Mat &img,
		const cv::Mat &mask, const std::vector<cv::Vec3d> &sparse, cv::Mat &depthmap,
		std::vector<cv::Vec3d> &dense, std::vector<cv::Vec3b> &color);

	/**
	 * @brief Checks if the pipeline is idling
	*/
	bool isIdle();

	/**
	 * @brief Number of frames needed before the first depth output
	*/
	uint8_t numPrerun();

	/**
	 * @brief Number of frames needed to initialize the pipeline without generating explicit output
	*/
	uint8_t numDropped();

	/**
	 * @brief Resets the pipeline
	*/
	void reset();

private:
	bool alive = true;
	double min_depth = 1e-5;
	int16_t count;
	bool initializing = true;
	double min_stereo_score = 0.25;
	int16_t num_drop = 1; // 1 is the calculation baseed of the default values in depthmap.cpp
	int16_t num_init = -2; // -2 is the same

	std::mutex m_input;
	std::queue<std::shared_ptr<DepthmapInput>> q_input;
	std::vector<cv::Mat> v_img1;
	std::vector<cv::Mat>::iterator i_img1;
	std::mutex m_estimator;
	std::unique_ptr<DepthmapEstimator> estimator;
	std::thread t_estimator;
	std::mutex m_hidden1;
	std::queue<DepthmapEstimatorResult*> q_hidden1;
	std::vector<cv::Mat> v_img2;
	std::vector<cv::Mat>::iterator i_img2;
	std::mutex m_cleaner;
	std::unique_ptr<DepthmapCleaner> cleaner;
	std::thread t_cleaner;
	std::mutex m_hidden2;
	std::queue<DepthmapCleanerResult*> q_hidden2;
	std::mutex m_pruner;
	std::unique_ptr<DepthmapPruner> pruner;
	std::thread t_pruner;
	std::mutex m_output;
	std::queue<DepthmapPrunerResult*> q_output;

	void run_estimator();
	void run_cleaner();
	void run_pruner();

	static double compute_max_depth_from_sparse(const cv::Matx33d &R, const cv::Vec3d &t,
		const std::vector<cv::Vec3d> &sparse);

	static bool compare_points(const cv::Vec3d *p1, const cv::Vec3d *p2);

	static double score_stereo(const cv::Vec3d &t1, const std::vector<cv::Vec3d> &sparse1,
		const cv::Vec3d &t2, const std::vector<cv::Vec3d> &sparse2);
};

}
