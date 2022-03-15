let CPM = require("../../artistoo/build/artistoo-cjs.js")
//let FPS = require("./fpsmeter.min.js")

let config = {

    // Grid settings
    ndim: 2,
    field_size: [210, 210],

    // CPM parameters and configuration
    conf: {
        // Basic CPM parameters
        torus: [true, true],			    // Should the grid have linked borders?
        seed: 1,							// Seed for random number generation.


        T: 20,								// CPM temperature

        // ##########################################################################
        // The values here configure adhesion, volume, perimeter and activity.
        // ##########################################################################
        J: [[0, 500, 20],
            [500, 500, 500],
            [20, 500, 20]],

        LAMBDA_V: [0, 500, 50],
        V: [0, 250, 500],

        LAMBDA_P: [0, 500, 2],
        P: [0, 20, 340],

        LAMBDA_ACT: [0, 0, 200],
        MAX_ACT: [0, 20, 80],
        ACT_MEAN: 'geometric'

    },

    // Simulation setup and configuration
    simsettings: {

        // ##########################################################################
        // Change the first value of NRCELLS to change the number of obstacle cells.
        // And the second number to change the number of moving cells.
        // ##########################################################################
        NRCELLS: [10, 10],					// Number of cells to seed for all
        CELLCOLOR: ["AAAAAA", "000000"],
        ACTCOLOR: [false, true],
        SHOWBORDERS: [true, false],
        // non-background cellkinds.

        RUNTIME: 5000,
        RUNTIME_BROWSER: 'Inf',

        // Visualization
        CANVASCOLOR: "eaecef",
        zoom: 2,							// zoom in on canvas with this factor.

        // Save the results
        SAVEIMG: true,
        IMGFRAMERATE: 5,
        SAVEPATH: "./assignment3_outputs/high_bc_bg",
        EXPNAME: "high_bc_bg"
    }
}
/*	---------------------------------- */

function initialize() {

    let custommethods = {
        initializeGrid: initializeGrid,
    }

    sim = new CPM.Simulation(config, custommethods)


    //meter = new FPS.FPSMeter({left: "auto", right: "5px"})
    step()
}


function step() {
    // sim.step()
    // meter.tick()
    for( let s = 0; s < config.simsettings.IMGFRAMERATE; s++ ){
        sim.step()
    }
    //meter.tick()

    if (sim.conf["RUNTIME_BROWSER"] == "Inf" | sim.time + 1 < sim.conf["RUNTIME_BROWSER"]) {
        requestAnimationFrame(step)
    }
}


function initializeGrid() {
    // add the GridManipulator if not already there and if you need it
    if (!this.helpClasses["gm"]) {
        this.addGridManipulator()
    }

    // CHANGE THE CODE BELOW TO FIT YOUR SIMULATION

    let nrcells = this.conf["NRCELLS"], cellkind, i

    // Seed the right number of cells for each cellkind
    for (cellkind = 0; cellkind < nrcells.length; cellkind++) {
        for (i = 0; i < nrcells[cellkind]; i++) {
            if (cellkind + 1 == 1) {
                // ##########################################################################
                // This seedCellAt call defines where to place the obstacles.
                // The second parameter is a array containing the x,y coordinates for the one specific cell.
                // ##########################################################################
                // this.gm.seedCellAt(cellkind + 1, [i * (200 / (nrcells[cellkind] - 1)), 100])
                this.gm.seedCell(cellkind + 1)
            } else {
                this.gm.seedCell(cellkind + 1)
            }
        }
    }
}


let sim = new CPM.Simulation( config )
//let meter
sim.run()