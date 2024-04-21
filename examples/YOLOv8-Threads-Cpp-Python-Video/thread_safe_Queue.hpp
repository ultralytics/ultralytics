/*
#############
###ZouJiu
###20240421
###1069679911@qq.com
#https://zoujiu.blog.csdn.net/
#https://zhihu.com/people/zoujiu1
#https://github.com/ZouJiu1
#############
*/

#ifndef THREAD_SAFE_QUEUE_HPP
#define THREAD_SAFE_QUEUE_HPP

#include <queue>
#include <condition_variable>
#include <mutex>
#include <iostream>

template<typename T>
class ThreadQueue : public std::queue<T>
{
private:
	std::mutex mut;
	std::condition_variable condition;
	std::queue<T> queue_data;
	int maxsize = -1;
public:
	int unfinished_tasks = 0;
	int finished_tasks = 0;
	ThreadQueue() {}
	~ThreadQueue() {}
	ThreadQueue(int max_size) : maxsize(max_size) {
	}
	void task_done() {
		std::lock_guard<std::mutex> lock_(mut);
		int unfinished = unfinished_tasks - 1;
		if (unfinished <= 0) {
			if (unfinished < 0) {
				std::cerr << "error, task_done() called too many times, exit -1." << std::endl;
				exit(-1);
			}
			//condition.notify_one();
		}
		unfinished_tasks = unfinished;
	}
	void join() {
		std::unique_lock<std::mutex> lock_(mut);
		condition.wait(lock_, [this] {return unfinished_tasks == 0; });
	}
	int qsize() {
		std::lock_guard<std::mutex> lock_(mut);
		return queue_data.size();
	}
	T front() {
		std::unique_lock<std::mutex> lock_(mut);
		condition.wait(lock_, [this] {return !empty(); });
		return queue_data.front();
	}
	bool empty() {
		return queue_data.size() == 0;
	}
	bool full() {
		return this->maxsize >= 0 && this->maxsize <= queue_data.size();
	}
	void put(T&& a) {
		std::unique_lock<std::mutex> lock_(mut);
		condition.wait(lock_, [this] {return !full(); });
		queue_data.push(a);
		unfinished_tasks++;
		condition.notify_one();
	}
	void get(T& a) {
		std::unique_lock<std::mutex> lock_(mut);
		condition.wait(lock_, [this] {return !empty(); });
		a = queue_data.front();
		queue_data.pop();
		condition.notify_one();
	}
};

#endif