env = 'http://127.0.0.1:5000'
var n = 7
var m = 7

main();

async function main() {
  board = await getEmptyBoard(env, n, m)
  displayBoard(board)

  // Just testing this
  await postBoard(board)
}

async function postBoard(board) {
  await fetch(`${env}/play`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(board)
  })
}

async function getRandomBoard(env, n, m) {
  return (await fetch(`${env}/board/random/${n}x${m}/`).then(response => { return response.json() })).board
}

async function getEmptyBoard(env, n, m) {
  return (await fetch(`${env}/board/empty/${n}x${m}/`).then(response => { return response.json() })).board
}

function displayBoard(board) {

  var n = board.length
  var m = board[0].length

  let cellSize = 100 - Math.min(((Math.max(n, m) - 3) * 8), 50)
  document.documentElement.style.setProperty("--cell-size", `${cellSize}px`)

  boardHtml = document.getElementById('board')

  boardHtml.style.setProperty("grid-template-columns", `repeat(${m}, 1fr)`)

  boardHtml.style.setProperty("width", `${m} * var(--cell-size) + ${m-1} * var(--gap-size)`)
  boardHtml.style.setProperty("height", `${n} * var(--cell-size) + ${n-1} * var(--gap-size)`)

  let cellType = [" O", "", " X"]
  let display = ""

  for (var i = 0; i < n; i++)
    for (var j = 0; j < m; j++)
      display += `<div class="cell${cellType[board[i][j] + 1]}" data-cell></div>`

  boardHtml.innerHTML = display
}