// The MIT License (MIT)
//
// Copyright (c) 2019 Luigi Pertoldi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//
// ============================================================================
//  ___   ___   ___   __    ___   ____  __   __   ___    __    ___
// | |_) | |_) / / \ / /`_ | |_) | |_  ( (` ( (` | |_)  / /\  | |_)
// |_|   |_| \ \_\_/ \_\_/ |_| \ |_|__ _)_) _)_) |_|_) /_/--\ |_| \_
//
// Very simple progress bar for c++ loops with size_ternal running variable
//
// Author: Luigi Pertoldi
// Created: 3 dic 2016
//
// Notes: The bar must be used when there's no other possible source of output
//        inside the for loop
//

#ifndef __PROGRESSBAR_HPP
#define __PROGRESSBAR_HPP

#include <iostream>
#include <string>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <atomic>

class progressbar {

    public:
      // default destructor
      inline ~progressbar()                             = default;

      // delete everything else
      inline progressbar           (progressbar const&) = delete;
      inline progressbar& operator=(progressbar const&) = delete;
      inline progressbar           (progressbar&&)      = delete;
      inline progressbar& operator=(progressbar&&)      = delete;

      // default constructor, must call set_niter later
      inline progressbar();
      inline progressbar(size_t n, bool showbar=true, bool init_update = true);

      // reset bar to use it again
      inline void reset();
     // set number of loop iterations
      inline void set_niter(size_t iter);
      // chose your style
      inline void set_done_char(const std::string& sym) {done_char = sym;}
      inline void set_todo_char(const std::string& sym) {todo_char = sym;}
      inline void set_opening_bracket_char(const std::string& sym) {opening_bracket_char = sym;}
      inline void set_closing_bracket_char(const std::string& sym) {closing_bracket_char = sym;}
      // to show only the percentage
      inline void show_bar(bool flag = true) {do_show_bar = flag;}
      // main function
      inline std::string get_update_str();

      inline virtual void update(size_t inc = 1);
    protected:
      std::atomic<size_t> progress;
      size_t n_cycles;
      std::atomic<bool> update_is_called;
    private:
      std::atomic<size_t> last_perc;
      bool do_show_bar;

      std::string done_char;
      std::string todo_char;
      std::string opening_bracket_char;
      std::string closing_bracket_char;
      std::chrono::time_point<std::chrono::high_resolution_clock> last_timepoint;
};

progressbar::progressbar() :
    progress(0),
    n_cycles(0),
    last_perc(0),
    do_show_bar(true),
    update_is_called(false),
    done_char("█"),
    todo_char(" "),
    opening_bracket_char("["),
    closing_bracket_char("]"),
    last_timepoint(std::chrono::high_resolution_clock::now()) {
        update();
    }

progressbar::progressbar(size_t n, bool showbar, bool init_update) :
    progress(0),
    n_cycles(n),
    last_perc(0),
    do_show_bar(showbar),
    update_is_called(false),
    done_char("█"),
    todo_char(" "),
    opening_bracket_char("["),
    closing_bracket_char("]"),
    last_timepoint(std::chrono::high_resolution_clock::now()) {
        if (init_update) {
            update();
        }
    }

void progressbar::reset() {
    progress = 0,
    update_is_called = false;
    last_perc = 0;
    update();
    last_timepoint = std::chrono::high_resolution_clock::now();
    return;
}

void progressbar::set_niter(size_t niter) {
    if (niter <= 0) throw std::invalid_argument(
        "progressbar::set_niter: number of iterations null or negative");
    n_cycles = niter;
    return;
}

void progressbar::update(size_t inc) {
    if (update_is_called) progress += inc;
    std::cerr<<get_update_str();
}

std::string progressbar::get_update_str() {
    std::stringstream progressstream;

    if (n_cycles == 0) throw std::runtime_error(
            "progressbar::update: number of cycles not set");
    if (do_show_bar)
        progressstream<<"\r";

    size_t perc = 0;
    // compute percentage, if did not change, do nothing and return
    perc = static_cast<size_t>(progress*100./static_cast<double>(n_cycles));

    if (progress >= n_cycles) {
        perc = 100;
    }
    
    // if (perc < last_perc) return;
    
    // update percentage each unit
    if ((perc > last_perc) || (!update_is_called)) {

        auto now = std::chrono::high_resolution_clock::now();
        // update bar every ten units
        if (do_show_bar) {

            progressstream<<opening_bracket_char;

            for (size_t j = 0;  j < 20; j++) {
                if (5*j + 1 < perc) {
                    progressstream<<done_char;
                } else {
                    progressstream<<todo_char;
                }
            }

            // readd trailing percentage characters
            progressstream << closing_bracket_char;

            if (perc < 10) {
                progressstream <<"  "<< perc << '%';
            } else if (perc  >= 10 && perc < 100) {
                progressstream <<" "<<perc << '%';
            } else {
                progressstream << 100 <<'%';
            }
        }

        std::string total = std::to_string(n_cycles);
        std::string progressstr = std::to_string(progress);

        int sizediff = total.size() - progressstr.size();

        if (do_show_bar) {
            progressstream<<"("<<std::string(sizediff,*const_cast<char*>(" "))
                <<progressstr << "/" << total <<")";
            progressstream << "  " << opening_bracket_char;
        }
        
        double sec = std::chrono::duration_cast<std::chrono::milliseconds>(now-last_timepoint).count() / 1000.0;
        double iterations_per_second = static_cast<double>(progress) / sec;

        if (do_show_bar) {
            progressstream << std::to_string(iterations_per_second).substr(0,7);
            progressstream << "it/s" << closing_bracket_char;
            if (perc == 100) {
                progressstream << std::endl;
            }
        }
    }
    if (!update_is_called) update_is_called = true;
    if (perc > last_perc) last_perc = perc;

    //progressstream << std::flush;
    return progressstream.str();
}

#endif
