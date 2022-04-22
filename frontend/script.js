var cells, board, winningMessage, winningMessageText, newGameButton, xIsNext;

// I wrote this terrible hacky code in this way because newGame() MUST trigger AFTER displayBoard()
// has fully finished and all elements have been loaded onto the page, since it operates on those elements.
// Adding a forced delay of 0.5s was just the easiest (but def not best) way to make it work
window.onpageshow = (event) => {
    setTimeout(() => {
        console.log("Page has been shown");
        newGame();
        newGameButton.addEventListener("click", newGame);
    }, 500);
};

// Change size of board based on user input
document.getElementById("start").addEventListener("click", () => {
    changeMNK(
        document.getElementById("m").value,
        document.getElementById("n").value,
        document.getElementById("k").value
    );
});

function updateBoard(board) {
    displayBoard(board);

    cells = document.querySelectorAll(".cell");

    cells.forEach((cell) => {
        cell.addEventListener("click", handleClick, { once: true });
    });
}

function newGame() {
    socket.emit("new_game", { m: m, n: n, k: k });
    console.log("Page has been shown 2");
    cells = document.querySelectorAll(".cell");
    board = document.getElementById("board");
    winningMessage = document.querySelector(".winningMessage");
    winningMessageText = document.querySelector(".winningMessageText");
    newGameButton = document.querySelector(".newGameButton");
    xIsNext = true;

    board.classList.add("X");
    board.classList.remove("O");
    winningMessage.classList.remove("show");

    cells.forEach((cell) => {
        cell.classList.remove("X");
        cell.classList.remove("O");
        cell.addEventListener("click", handleClick, { once: true });
    });
}

function getCoordinates(str) {
    // Given a string in the form "i, j" return the coordinates as an array [i, j]
    var coordinates = str.split(",");
    return [parseInt(coordinates[0]), parseInt(coordinates[1])];
}

async function changeMNK(m, n, k) {
    board = await getEmptyBoard(env, m, n);
    displayBoard(board);
    newGame();

    // Post to server signal that new n, m, k have been chosen
    socket.emit("new_game", { m: m, n: n, k: k });
}

function handleClick(e) {
    // place mark
    const cell = e.target;
    const player = xIsNext ? "X" : "O";
    cell.classList.add(player);

    // check for draw
    if (
        [...cells].every((cell) => {
            return cell.classList.contains("X") || cell.classList.contains("O");
        })
    ) {
        winningMessageText.innerText = `It's a draw.`;
        winningMessage.classList.add("show");
    }
    // switch turns
    else {
        // xIsNext = !xIsNext;
        if (xIsNext) {
            board.classList.add("X");
            board.classList.remove("O");
        } else {
            board.classList.add("O");
            board.classList.remove("X");
        }
    }

    // Post to move to server. Send coordinates previously encoded in cell ID.
    const coordinates = getCoordinates(e.target.getAttribute("id"));
    socket.emit("user_move", { i: coordinates[0], j: coordinates[1] });
}

// check for win
function displayWinningMessage(who_won) {
    winningMessageText.innerText = `${who_won ? "X" : "O"} Wins!`;
    winningMessage.classList.add("show");
}

// Receives board from server and displays it
socket.on("board_update", (board) => {
    updateBoard(board);
});

socket.on("win", (who_won) => {
    displayWinningMessage(who_won);
});

// TODO
function checkForWin(player) {
    return false;

    /*
	var m = 7, n = 7
	var k_in_a_row = 3

	for (var r = 0; r < m; r++)
		for (var c = 0; c < n; c++)
			if (winFromPos(r, c, k_in_a_row))
				return true
	*/
}

// TODO
function winFromPos(r, c, k_in_a_row) {
    /*
	var m = 7, n = 7
	var k_in_a_row = 3

	cells[index].classList.contains(player)
	*/
}
