<script src="./models.js"></script>
// var iter = mr.pixelToComplex(x, y);
// var orig = iter;
// var iteration = 0;
// while(iteration < maxIterations && iter.real**2 + iter.imaginary**2 <= 4) {
//     iteration+=1;
//     iter = iter.iterate(orig);
// }
// var p = mr.complexToPixel(orig);


// return {
//     iterations: iterations,
//     x: x,
//     y: y
// }

class ComplexNumber {
    constructor(real, imaginary) {
        this.real = real;
        this.imaginary = imaginary;
    }

    add(cn) {
        return new ComplexNumber(
            this.real + cn.real,
            this.imaginary + cn.imaginary
        )
    }

    mult(cn) {
        return new ComplexNumber(
            (this.real * cn.real) - (this.imaginary * cn.imaginary),
            (this.real * cn.imaginary) + (this.imaginary * cn.real)
        )
    }

    square() {
        return this.mult(this)
    }

    iterate(orig) {
        return this.square().add(orig);
    }

    toString() {
        return this.real + " + " + this.imaginary + "i";
    }
}

self.addEventListener('message', function(e) {
    if(parseInt(e.data.x) % 100 == 0 && parseInt(e.data.y) % 100 == 0) {
        console.log(e.data.mr(e.data.x, e.data.y));
    }
})