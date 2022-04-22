env = "http://127.0.0.1:5000";
var m = 7;
var n = 3;

main();

async function main() {
    board = await getEmptyBoard(env, m, n);
    displayBoard(board);

    // Just testing this
    await postBoard(board);
}

async function postBoard(board) {
    await fetch(`${env}/play`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(board),
    });
}

async function getRandomBoard(env, m, n) {
    return (
        await fetch(`${env}/board/random/${m}x${n}/`).then((response) => {
            return response.json();
        })
    ).board;
}

async function getEmptyBoard(env, m, n) {
    return (
        await fetch(`${env}/board/empty/${m}x${n}/`).then((response) => {
            return response.json();
        })
    ).board;
}

function displayBoard(board) {
    var m = board.length;
    var n = board[0].length;

    let cellSize = 100 - Math.min((Math.max(m, n) - 3) * 8, 50);
    document.documentElement.style.setProperty("--cell-size", `${cellSize}px`);

    boardHtml = document.getElementById("board");

    boardHtml.style.setProperty("grid-template-columns", `repeat(${n}, 1fr)`);

    boardHtml.style.setProperty("height", `${m} * var(--cell-size) + ${m - 1} * var(--gap-size)`);
    boardHtml.style.setProperty("width", `${n} * var(--cell-size) + ${n - 1} * var(--gap-size)`);

    let cellType = [" O", "", " X"];
    let display = "";

    for (var i = 0; i < m; i++)
        for (var j = 0; j < n; j++)
            display += `<div id="${i}, ${j}" class="cell${
                cellType[board[i][j] + 1]
            }" data-cell></div>`;

    boardHtml.innerHTML = display;
}
