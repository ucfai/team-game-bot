var n = 7
var m = 7

displayBoard(n, m);

async function displayBoard(n, m) {

  var input_board_test = (await fetch(`https://team-game-bot.herokuapp.com/board/${n}x${m}/`).then(response => { return response.json() })).board

  var n = input_board_test.length
  var m = input_board_test[0].length

  let cellSize = 100 - Math.min(((Math.max(n, m) - 3) * 8), 50)
  document.documentElement.style.setProperty("--cell-size", `${cellSize}px`);

  document.getElementById('board').style.setProperty("grid-template-columns", `repeat(${m}, 1fr)`);

  document.getElementById('board').style.setProperty("width", `${m} * var(--cell-size) + ${m-1} * var(--gap-size)`);
  document.getElementById('board').style.setProperty("height", `${n} * var(--cell-size) + ${n-1} * var(--gap-size)`);

  let cellType = [" O", "", " X"]
  let display = ""

  for (var i = 0; i < n; i++)
    for (var j = 0; j < m; j++)
      display += `<div class="cell${cellType[input_board_test[i][j] + 1]}" data-cell></div>`

  document.getElementById('board').innerHTML = display;
}