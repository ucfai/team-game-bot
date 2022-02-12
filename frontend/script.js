const WINNING_COMBINATIONS = [
	[0, 1, 2],
	[3, 4, 5],
	[6, 7, 8],
	[0, 3, 6],
	[1, 4, 7],
	[2, 5, 8],
	[0, 4, 8],
	[2, 4, 6],
]
const cellElements = document.querySelectorAll('[data-cell]')
const board = document.getElementById('board')
const winningMessage = document.getElementById('winningMessage')
const winningMessageText = document.querySelector('[data-winning-message-text]')
const restartButton = document.getElementById('restartButton')
let circleTurn = false

restartButton.addEventListener('click', restart)

function restart() {
	circleTurn = false
	board.classList.remove('circle')
	board.classList.add('x')
	winningMessage.classList.remove('show')
	cellElements.forEach(cell => {
		cell.classList.remove('x')
		cell.classList.remove('circle')
		cell.addEventListener('click', handleClick, {once: true})
	})
}

cellElements.forEach(cell => {
	cell.addEventListener('click', handleClick, {once: true})
})

function handleClick(e) {
	// place mark
	const cell = e.target
	const currentClass = circleTurn ? 'circle' : 'x'
	cell.classList.add(currentClass)
	// check for win
	if (WINNING_COMBINATIONS.some(combination => {
		return combination.every(index => {
			return cellElements[index].classList.contains(currentClass)
		})
	})) {
		winningMessageText.innerText = `${circleTurn ? "O" : "X"} Wins!`
		winningMessage.classList.add('show')
	}
	// check for draw
	else if ([...cellElements].every(cell => {
		return cell.classList.contains('x') || cell.classList.contains('circle')
	})) {
		winningMessageText.innerText = `It's a draw.`
		winningMessage.classList.add('show')
	}
	// switch turns
	else
	{
		circleTurn = !circleTurn
		if (circleTurn) {
			board.classList.remove('x')
			board.classList.add('circle')
		} else {
			board.classList.remove('circle')
			board.classList.add('x')
		}
	}
}
