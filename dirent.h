/* dirent.h for Windows */
#ifndef DIRENT_H
#define DIRENT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <windows.h>
#include <stdio.h>

#define MAX_FILENAME_LEN 260

    struct dirent {
        char d_name[MAX_FILENAME_LEN];  // 파일 이름
    };

    typedef struct {
        HANDLE hFind;
        WIN32_FIND_DATAA findFileData;
        struct dirent dirent;
        char search_path[MAX_FILENAME_LEN];
        int first;
    } DIR;

    DIR* opendir(const char* name);
    struct dirent* readdir(DIR* dirp);
    int closedir(DIR* dirp);

#ifdef __cplusplus
}
#endif

#endif /* DIRENT_H */
