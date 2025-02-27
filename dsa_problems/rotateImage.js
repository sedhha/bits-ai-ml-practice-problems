function rotateMatrix(matrix, rotation) {
    let m = matrix.length, n = matrix[0].length;

    // Normalize rotation (convert negative to equivalent positive)
    rotation = ((rotation % 360) + 360) % 360;

    // If 360° (or 0°), return the original matrix
    if (rotation === 0) return matrix;

    let rotated;

    if (rotation === 90) {
        rotated = Array.from({ length: n }, () => Array(m));
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                rotated[j][m - 1 - i] = matrix[i][j]; // Transpose + Reverse rows
            }
        }
    } else if (rotation === 180) {
        rotated = Array.from({ length: m }, () => Array(n));
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                rotated[m - 1 - i][n - 1 - j] = matrix[i][j]; // Reverse rows + Reverse columns
            }
        }
    } else if (rotation === 270) {
        rotated = Array.from({ length: n }, () => Array(m));
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                rotated[n - 1 - j][i] = matrix[i][j]; // Transpose + Reverse columns
            }
        }
    }

    return rotated;
}

// Example Usage
let matrix = [
    [1, 2, 3],
    [4, 5, 6]
];

console.log("90° Rotation:");
console.log(rotateMatrix(matrix, 90));

console.log("180° Rotation:");
console.log(rotateMatrix(matrix, 180));

console.log("270° Rotation:");
console.log(rotateMatrix(matrix, 270));

console.log
