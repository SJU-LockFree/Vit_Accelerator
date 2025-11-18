#include "dirent.h"
#include <string.h>
#include <stdlib.h>
#define _CRT_SECURE_NO_WARNINGS

DIR* opendir(const char* name) {
    DIR* dir = (DIR*)malloc(sizeof(DIR));
    if (!dir) return NULL;

    snprintf(dir->search_path, MAX_FILENAME_LEN, "%s\\*", name);
    dir->hFind = FindFirstFileA(dir->search_path, &dir->findFileData);
    if (dir->hFind == INVALID_HANDLE_VALUE) {
        free(dir);
        return NULL;
    }

    dir->first = 1;
    return dir;
}

struct dirent* readdir(DIR* dirp) {
    if (!dirp) return NULL;

    if (dirp->first) {
        dirp->first = 0;
    }
    else {
        if (!FindNextFileA(dirp->hFind, &dirp->findFileData))
            return NULL;
    }

    strncpy_s(dirp->dirent.d_name, MAX_FILENAME_LEN, dirp->findFileData.cFileName, _TRUNCATE);
    dirp->dirent.d_name[MAX_FILENAME_LEN - 1] = '\0'; // null-terminate

    return &dirp->dirent;
}

int closedir(DIR* dirp) {
    if (!dirp) return -1;
    FindClose(dirp->hFind);
    free(dirp);
    return 0;
}
