var cells, board, winningMessage, winningMessageText, newGameButton, xIsNext

// I wrote this terrible hacky code in this way because newGame() MUST trigger AFTER displayBoard()
// has fully finished and all elements have been loaded onto the page, since it operates on those elements.
// Adding a forced delay of 0.5s was just the easiest (but def not best) way to make it work
window.onpageshow = (event) => {
	setTimeout(() => {
		newGame()
		newGameButton.addEventListener('click', newGame)
	}, 500);
};

function newGame() {
	cells = document.querySelectorAll('.cell')
	board = document.getElementById('board')
	winningMessage = document.querySelector('.winningMessage')
	winningMessageText = document.querySelector('.winningMessageText')
	newGameButton = document.querySelector('.newGameButton')
	xIsNext = true

	board.classList.add('X')
	board.classList.remove('O')
	winningMessage.classList.remove('show')

	cells.forEach(cell => {
		cell.classList.remove('X')
		cell.classList.remove('O')
		cell.addEventListener('click', handleClick, {once: true})
	})
}

function handleClick(e) {
	// place mark
	const cell = e.target
	const player = xIsNext ? 'X' : 'O'
	cell.classList.add(player)
	// check for win
	if (checkForWin(player)) {
		winningMessageText.innerText = `${xIsNext ? 'X' : 'O'} Wins!`
		winningMessage.classList.add('show')
	}
	// check for draw
	else if ([...cells].every(cell => {
		return cell.classList.contains('X') || cell.classList.contains('O')
	})) {
		winningMessageText.innerText = `It's a draw.`
		winningMessage.classList.add('show')
	}
	// switch turns
	else
	{
		xIsNext = !xIsNext
		if (xIsNext) {
			board.classList.add('X')
			board.classList.remove('O')
		} else {
			board.classList.add('O')
			board.classList.remove('X')
		}
	}
}

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
