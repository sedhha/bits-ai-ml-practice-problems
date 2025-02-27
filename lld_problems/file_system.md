# Low-Level Design (LLD) for a File System

## 1. Objective

Create a simplified file system to manage files and directories.

## 2. Key Considerations

- **Hierarchy Management:** Maintain a tree-like structure for files and directories.
- **File Operations:** Support creation, deletion, moving, and reading of files.
- **Permission Handling:** Implement user-based access control (read, write, execute).
- **Storage Optimization:** Efficiently manage disk space.

## 3. Class Design

### 3.1 Key Classes

```java
class FileSystem {
    private Directory root;

    public FileSystem() {
        this.root = new Directory("/");
    }

    public boolean createFile(String path, String name, String content);
    public boolean createDirectory(String path, String name);
    public boolean delete(String path);
    public boolean move(String sourcePath, String destinationPath);
    public File getFile(String path);
    public Directory getDirectory(String path);
}

class Directory {
    private String name;
    private List<File> files;
    private List<Directory> subDirectories;
    private Directory parent;

    public Directory(String name);
    public boolean addFile(File file);
    public boolean addDirectory(Directory directory);
    public boolean remove(String name);
}

class File {
    private String name;
    private String content;
    private int size;
    private Directory parent;
    private Permissions permissions;

    public File(String name, String content);
    public void writeContent(String content);
    public String readContent();
}

class Permissions {
    private boolean read;
    private boolean write;
    private boolean execute;

    public Permissions(boolean read, boolean write, boolean execute);
}
```

## 4. Database Schema (For Persistent Storage)

### 4.1 Files Table

```sql
CREATE TABLE Files (
    file_id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    content TEXT,
    size INT,
    directory_id INT REFERENCES Directories(directory_id),
    read_permission BOOLEAN DEFAULT TRUE,
    write_permission BOOLEAN DEFAULT TRUE,
    execute_permission BOOLEAN DEFAULT FALSE
);
```

### 4.2 Directories Table

```sql
CREATE TABLE Directories (
    directory_id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    parent_id INT REFERENCES Directories(directory_id)
);
```

## 5. System Workflow

### 5.1 File System Operations

1. **Create File:**
   - Locate the target directory.
   - Check permissions.
   - Create a new file entry.
2. **Create Directory:**
   - Locate parent directory.
   - Check if directory exists.
   - Create a new directory entry.
3. **Delete Operation:**
   - Remove file or directory if permissions allow.
4. **Move Operation:**
   - Validate paths.
   - Update parent directory reference.
5. **Read File Content:**
   - Validate permissions.
   - Return content.

## 6. Optimizations and Trade-offs

- **In-Memory Caching:** Cache frequently accessed directories.
- **Lazy Loading:** Load directory contents only when accessed.
- **Concurrency Control:** Implement file locking for safe multi-user access.
- **Space Management:** Use compression and deduplication for efficient storage.

## 7. Conclusion

This LLD provides a structured way to implement a **File System**, covering class design, database schema, and operations workflow. The design ensures efficient file and directory management while enforcing access control and storage optimization.
