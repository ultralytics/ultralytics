// 动态库设置
// 当windows时设置宏为动态库
// 当为Linux时设置宏为none
#ifndef CONFIG_H
#define CONFIG_H
    #ifdef _WIN32
        #define CPP_API __declspec(dllexport)
    #else
        #define CPP_API 
    #endif
#endif