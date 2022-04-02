const cells = document.querySelectorAll('.cell')
var board = document.querySelector('.board')
const winningMessage = document.querySelector('.winningMessage')
const winningMessageText = document.querySelector('.winningMessageText')
const newGameButton = document.querySelector('.newGameButton')
let xIsNext

newGame()

newGameButton.addEventListener('click', newGame)

function newGame() {
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

const lines = [
	[0, 1, 2],
	[3, 4, 5],
	[6, 7, 8],
	[0, 3, 6],
	[1, 4, 7],
	[2, 5, 8],
	[0, 4, 8],
	[2, 4, 6],
]

function handleClick(e) {
	// place mark
	const cell = e.target
	const player = xIsNext ? 'X' : 'O'
	cell.classList.add(player)
	// check for win
	if (lines.some(combination => {
		return combination.every(index => {
			return cells[index].classList.contains(player)
		})
	})) {
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
